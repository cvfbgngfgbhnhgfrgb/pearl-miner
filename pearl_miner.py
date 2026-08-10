#!/usr/bin/env python3
"""
pearl_miner.py — Pearl (PRL) "PearlHash" miner (the worker process).

What it does
------------
1. Polls jobs.txt on GitHub (written by the pool connector).
2. For every new job it mines with the GPU (PyTorch) — the actual work is
   int8 matrix multiplication (the "useful work" of PearlHash): it builds
   signal matrices A/B, adds low-rank noise, computes the noisy GEMM on the
   GPU and accumulates per-tile XOR "jackpot" transcripts, then checks
   blake3(jackpot, key=sigma) <= target.
3. When a tile passes the target, the share (with its merkle proof) is
   appended to shares.txt and pushed to GitHub. The pool connector picks it
   up, submits it to the pool and clears the file.

Usage
-----
  GH_TOKEN=ghp_... python pearl_miner.py --rig rig1
  python pearl_miner.py --bus local --local-dir /tmp/bus --rig rig1
  # relax the target for a demo (real pool target is ~2^203 — too hard for
  # a Python/PyTorch miner; see README):
  python pearl_miner.py --share-target-bits 248
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time

from pearlhash.commitment import hash_le32, jackpot_hash
from pearlhash.mining import JackpotSearcher, mine_once
from pearlhash.params import (
    IncompleteBlockHeader,
    MiningConfiguration,
    nbits_to_difficulty,
    penalized_target_bound,
)

log = logging.getLogger("miner")

DEFAULT_REPO = "pearl-miner"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pearl (PRL) PearlHash miner")
    p.add_argument("--repo", default=os.environ.get("PEARL_REPO", DEFAULT_REPO))
    p.add_argument("--bus", choices=["github", "local"], default="github")
    p.add_argument("--local-dir", default="/tmp/pearl_bus")
    p.add_argument("--rig", default=None, help="worker name (default: auto)")
    p.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--share-target-bits", type=int, default=None,
                   help="DEMO: force target = 2^bits (the real pool target is "
                        "~2^203 and will practically never be hit by a "
                        "Python/PyTorch miner)")
    p.add_argument("--max-attempts", type=int, default=0, help="0 = unlimited")
    p.add_argument("--poll-jobs", type=float, default=3.0)
    p.add_argument("--device", default=None, help="torch device override (cuda:0)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def make_bus(args):
    if args.bus == "local":
        from github_bus import LocalBus
        return LocalBus(args.local_dir)
    from github_bus import GitHubBus, get_token_from_env_or_config, resolve_repo
    token = get_token_from_env_or_config()
    repo = resolve_repo(args.repo, token)
    return GitHubBus(token, repo)


def resolve_target(job: dict, args) -> int | None:
    """Return the target to mine against, or None if unusable."""
    if args.share_target_bits is not None:
        t = 1 << args.share_target_bits
        log.warning("DEMO target override: 2^%d (pool target is %s)",
                    args.share_target_bits, (job.get("target") or "?")[:16])
        return t
    if job.get("target"):
        t = int(job["target"], 16)
        bits = t.bit_length()
        if bits < 240:
            log.warning(
                "Pool target is only 2^%d — a Python/PyTorch miner will "
                "practically never hit it. For a demo use "
                "--share-target-bits 248 instead.", bits)
        return t
    # fall back: nbits-derived, rank-penalized bound
    header = IncompleteBlockHeader.from_bytes(bytes.fromhex(job["header"]))
    config = MiningConfiguration(
        common_dim=args.k, rank=args.rank,
        rows_pattern=[0, 8],
        cols_pattern=[0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49,
                      56, 57, 64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 104, 105,
                      112, 113, 120, 121, 128, 129, 136, 137, 144, 145, 152,
                      153, 160, 161, 168, 169, 176, 177, 184, 185, 192, 193,
                      200, 201, 208, 209, 216, 217, 224, 225, 232, 233, 240,
                      241, 248, 249],
    )
    bound = penalized_target_bound(nbits_to_difficulty(header.nbits), config)
    if bound is None:
        log.warning("target too easy (would saturate); refusing to mine")
        return None
    return bound


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    rig = args.rig or f"{platform.node() or 'pc'}-pearl"
    bus = make_bus(args)
    log.info("rig=%s bus=%s backend=%s profile m=%d n=%d k=%d rank=%d",
             rig, type(bus).__name__, args.backend, args.m, args.n, args.k, args.rank)

    # backend + device
    if args.backend == "auto":
        try:
            import torch  # noqa: F401
            args.backend = "torch"
        except Exception:
            args.backend = "numpy"
    if args.backend == "torch":
        import torch
        dev = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("torch %s device=%s cuda_available=%s",
                 torch.__version__, dev, torch.cuda.is_available())
        if not torch.cuda.is_available():
            log.warning("no CUDA GPU found — running on CPU via torch")

    last_job_id = None
    attempts = 0
    total_hashes = 0
    total_time = 0.0
    started_at = time.time()

    def stats():
        rate = total_hashes / max(total_time, 1e-9)
        return f"{total_hashes} tile-hashes, {rate:,.0f} tile-hash/s"

    log.info("polling jobs.txt every %.1fs ...", args.poll_jobs)
    try:
        while True:
            text = bus.read_text("jobs.txt")
            if not text:
                time.sleep(args.poll_jobs)
                continue
            try:
                job = json.loads(text.strip().splitlines()[-1])
            except Exception:
                log.warning("bad jobs.txt content; waiting")
                time.sleep(args.poll_jobs)
                continue

            if job.get("job_id") == last_job_id:
                time.sleep(args.poll_jobs)
                continue

            last_job_id = job.get("job_id")
            header = IncompleteBlockHeader.from_bytes(bytes.fromhex(job["header"]))
            target = resolve_target(job, args)
            if target is None:
                log.warning("no usable target for job %s; waiting", job.get("job_id"))
                last_job_id = None
                time.sleep(args.poll_jobs)
                continue

            log.info("mining job %s height=%s target=%s...",
                     job.get("job_id"), job.get("height"), f"{target:064x}"[:16])

            config = MiningConfiguration(
                common_dim=args.k, rank=args.rank,
                rows_pattern=[0, 8],
                cols_pattern=[0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48,
                              49, 56, 57, 64, 65, 72, 73, 80, 81, 88, 89, 96,
                              97, 104, 105, 112, 113, 120, 121, 128, 129, 136,
                              137, 144, 145, 152, 153, 160, 161, 168, 169, 176,
                              177, 184, 185, 192, 193, 200, 201, 208, 209, 216,
                              217, 224, 225, 232, 233, 240, 241, 248, 249],
            )
            searcher = JackpotSearcher(config, args.m, args.n)
            log.info("per-attempt tiles: %d x %d = %d",
                     searcher.R, searcher.C, searcher.R * searcher.C)

            nonce = 0
            job_started = time.time()
            while True:
                # stop and switch if a newer job arrived
                new_text = bus.read_text("jobs.txt")
                if new_text:
                    try:
                        newer = json.loads(new_text.strip().splitlines()[-1])
                        if newer.get("job_id") != last_job_id:
                            log.info("newer job %s arrived; switching", newer.get("job_id"))
                            break
                    except Exception:
                        pass

                share = mine_once(header, config, args.m, args.n, target,
                                  nonce, backend=args.backend)
                nonce += 1
                attempts += 1
                if share:
                    from pearlhash.mining import share_to_line
                    line = share_to_line(share, rig, job.get("job_id"))
                    log.info("*** SHARE FOUND nonce=%s t=(%s,%s) jackpot=%s",
                             share["nonce"], share["t_rows"], share["t_cols"],
                             share["jackpot_hash"][:16])
                    bus.append_line("shares.txt", line)
                    log.info("share appended to shares.txt (%d bytes)", len(line))
                    total_hashes += share.get("hashes", 0)

                if args.max_attempts and attempts >= args.max_attempts:
                    log.info("reached --max-attempts=%d; stopping", args.max_attempts)
                    return
                if time.time() - started_at >= 30 and attempts % 5 == 0:
                    log.info("stats: attempts=%d %s", attempts, stats())

    except KeyboardInterrupt:
        log.info("bye")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\n✋ {e}", file=sys.stderr)
        sys.exit(1)
