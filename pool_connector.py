#!/usr/bin/env python3
"""
pool_connector.py — Pearl (PRL) pool connector (the "master" process).

What it does
------------
1. Asks how many PCs/workers are joining this mining session (or --workers N).
2. Connects to the Pearl stratum pool (default prl.kryptex.network:7048) —
   one authenticated connection per worker.
3. Receives mining.notify jobs from the pool, *rewrites* them into
   jobs.txt and pushes it to GitHub.
4. Watches shares.txt on GitHub. As soon as it is updated/rewritten by the
   miners it fetches the new shares, submits each to the pool
   (mining.submit), then clears shares.txt and waits for the next batch.

Usage
-----
  GH_TOKEN=ghp_... python pool_connector.py --workers 3
  python pool_connector.py                     # prompts for # of PCs
  python pool_connector.py --bus local --local-dir /tmp/bus   # offline test
  python pool_connector.py --mock localhost:19000             # mock pool test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time

from pearlhash.stratum import StratumClient, build_submit_params, parse_pool_url

log = logging.getLogger("connector")

DEFAULT_WALLET = "krxYRPV4WQ.0x"
DEFAULT_POOL = "prl.kryptex.network:7048"
DEFAULT_REPO = "pearl-miner"          # owner/repo or just repo (owner = token owner)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pearl (PRL) pool connector")
    p.add_argument("--pool", default=os.environ.get("PEARL_POOL", DEFAULT_POOL),
                   help=f"stratum endpoint (default {DEFAULT_POOL})")
    p.add_argument("--wallet", default=os.environ.get("PEARL_WALLET", DEFAULT_WALLET),
                   help=f"pool account id / wallet (default {DEFAULT_WALLET})")
    p.add_argument("--workers", type=int, default=None,
                   help="number of PCs/workers joining (default: ask)")
    p.add_argument("--worker-prefix", default="rig",
                   help="worker name prefix (rig1, rig2, ...)")
    p.add_argument("--repo", default=os.environ.get("PEARL_REPO", DEFAULT_REPO),
                   help=f"GitHub repo for the job/share bus (default {DEFAULT_REPO})")
    p.add_argument("--bus", choices=["github", "local"], default="github")
    p.add_argument("--local-dir", default="/tmp/pearl_bus",
                   help="directory for --bus local")
    p.add_argument("--mock", default=None,
                   help="use a mock pool at host:port instead of the real one")
    p.add_argument("--profile-m", type=int, default=1024, help="A rows (mining profile)")
    p.add_argument("--profile-n", type=int, default=1024, help="B cols (mining profile)")
    p.add_argument("--profile-k", type=int, default=1024, help="common dim (mining profile)")
    p.add_argument("--profile-rank", type=int, default=128, help="noise rank (mining profile)")
    p.add_argument("--poll-jobs", type=float, default=2.0, help="job write interval guard (s)")
    p.add_argument("--poll-shares", type=float, default=3.0, help="shares.txt poll interval (s)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def make_bus(args):
    if args.bus == "local":
        from github_bus import LocalBus
        return LocalBus(args.local_dir), args.local_dir
    from github_bus import GitHubBus
    repo = args.repo if "/" in args.repo else None
    if not repo:
        # resolve owner from token
        import requests
        token = os.environ.get("GH_TOKEN") or (json.load(open(_find_config()))
                                               .get("github_token", "") if os.path.exists(_find_config()) else "")
        me = requests.get("https://api.github.com/user",
                          headers={"Authorization": f"Bearer {token}"}).json()
        repo = f"{me['login']}/{args.repo}"
    bus = GitHubBus.from_env_or_config(repo)
    return bus, repo


def _find_config():
    for name in ("config.json", os.path.expanduser("~/.pearl_miner.json")):
        if os.path.exists(name):
            return name
    return "config.json"


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # ---- 1. how many PCs? ------------------------------------------------
    n_workers = args.workers
    if n_workers is None:
        try:
            n_workers = int(input("How many PCs are joining this mining session? "))
        except (EOFError, ValueError):
            n_workers = 1
    if n_workers < 1:
        n_workers = 1
    workers = [f"{args.wallet}.{args.worker_prefix}{i}" for i in range(1, n_workers + 1)]
    log.info("workers: %s", workers)

    bus, repo = make_bus(args)
    log.info("job/share bus: %s (repo=%s)", type(bus).__name__, repo)

    profile = {
        "m": args.profile_m, "n": args.profile_n, "k": args.profile_k,
        "rank": args.profile_rank,
        "rows_pattern": [0, 8],
        "cols_pattern": [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49,
                         56, 57, 64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 104,
                         105, 112, 113, 120, 121, 128, 129, 136, 137, 144, 145,
                         152, 153, 160, 161, 168, 169, 176, 177, 184, 185, 192,
                         193, 200, 201, 208, 209, 216, 217, 224, 225, 232, 233,
                         240, 241, 248, 249],
    }

    # ---- 2. pool connections (one per worker) -----------------------------
    host, port, tls = parse_pool_url(args.mock if args.mock else args.pool)
    clients = []
    for w in workers:
        c = StratumClient(host, port, w, tls=tls, on_job=None)
        clients.append(c)
        if args.mock:
            log.info("mock pool connection for %s", w)
        else:
            log.info("connecting %s -> %s:%s (tls=%s)", w, host, port, tls)
        c.connect()

    # ---- 3. job -> jobs.txt ----------------------------------------------
    last_written_job_id = None
    bus_lock = threading.Lock()

    def publish_job(job: dict):
        nonlocal last_written_job_id
        if job.get("job_id") == last_written_job_id:
            return
        payload = {
            "job_id": job.get("job_id"),
            "header": job.get("header"),
            "height": job.get("height"),
            "target": job.get("target"),
            "cert_version": job.get("cert_version"),
            "received_ts": job.get("received_ts"),
            "profile": profile,
        }
        with bus_lock:
            try:
                bus.write_text("jobs.txt", json.dumps(payload, separators=(",", ":")))
                last_written_job_id = job.get("job_id")
                log.info("published job %s (height=%s) -> jobs.txt",
                         job.get("job_id"), job.get("height"))
            except Exception as e:
                log.warning("failed to publish job %s: %s", job.get("job_id"), e)

    for c in clients:
        c.on_job = publish_job
    for c in clients:
        if c.job:
            publish_job(c.job)

    # ---- 4. shares.txt -> pool -------------------------------------------
    processed_lines = 0
    log.info("watching shares.txt every %.1fs ...", args.poll_shares)
    try:
        while True:
            time.sleep(args.poll_shares)
            content = bus.read_text("shares.txt")
            if not content or not content.strip():
                continue
            lines = [l for l in content.splitlines() if l.strip()]
            if len(lines) <= processed_lines:
                continue
            with bus_lock:
                for ln in new_lines:
                    try:
                        share = json.loads(ln)
                    except Exception:
                        log.warning("skipping unparseable share line: %.120s", ln)
                        continue
                    rig = share.get("rig", "rig1")
                    worker = f"{args.wallet}.{rig}"
                    client = next((c for c in clients if c.worker == worker), clients[0])
                    job = client.job or {}
                    params = build_submit_params(job, share)
                    log.info("submitting share rig=%s job=%s nonce=%s",
                             rig, share.get("job_id"), share.get("nonce"))
                    try:
                        resp = client.submit(params)
                        log.info("pool response: result=%s error=%s",
                                 resp.get("result"), resp.get("error"))
                    except Exception as e:
                        log.warning("submit failed: %s", e)
                # clear the file (keep only lines appended after our read)
                try:
                    bus.clear("shares.txt")
                    processed_lines = 0
                except Exception as e:
                    log.warning("clear shares.txt failed: %s", e)
    except KeyboardInterrupt:
        log.info("bye")
    finally:
        for c in clients:
            c.close()


if __name__ == "__main__":
    main()
