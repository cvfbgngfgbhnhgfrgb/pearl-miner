# 🐚 Pearl Miner — Pearl (PRL) PearlHash mining pipeline

A complete, working **Pearl (PRL)** mining setup built with **Python + PyTorch**,
using **GitHub as the job/share message bus** between the pool and your rigs.

```
                    ┌────────────────────────────────────────────┐
                    │              GitHub repo                    │
                    │         (private, token-protected)          │
                    │                                             │
   pool            │   jobs.txt  ◄──── written by the connector  │
 prl.kryptex ──►   │   shares.txt ────► read/drained by connector│
 .network:7048     │   shares.txt ◄──── appended by the miners   │
    ▲              │   jobs.txt  ────► read by the miners        │
    │              └────────────────────────────────────────────┘
    └── pool_connector.py  (master)         pearl_miner.py (worker, per PC)
```

* **pool_connector.py** — connects to the Pearl stratum pool, asks how many PCs
  are joining, rewrites each received job into `jobs.txt` on GitHub, watches
  `shares.txt`, submits new shares to the pool and clears the file.
* **pearl_miner.py** — runs on each PC: polls `jobs.txt`, mines with the GPU
  (PyTorch matmul — the actual PearlHash "useful work"), and appends found
  shares to `shares.txt` on GitHub.

---

## ⚠️ Read this first (honest engineering notes)

1. **Real-pool share format is pool-specific.** The Kryptex Pearl pool uses a
   custom stratum with **object-param `mining.submit`** (array params are
   rejected with `Unsupported submit format`) and a pool-specific proof
   encoding (Kryptex's own `krig` miner sends protobuf-encoded merkle proofs).
   This repo ships a fully documented **`v2-json` proof encoding** that the
   bundled mock pool verifies end-to-end. The connector's submit builder
   (`pearlhash/stratum.py: build_submit_params`) is a single pluggable
   function — swap in the pool's exact serialization there for production use.
2. **Python/PyTorch hashrate is tiny.** Real Pearl miners do ~100–300 TH/s
   with custom CUDA/HIP tensor-core kernels. A PyTorch miner does a few
   attempts/sec, so hitting the real pool target (`~2^203`) is practically
   impossible. Use `--share-target-bits N` for demos, or use the mock pool.
   This project is a **complete, correct, inspectable reference pipeline**,
   not a competitive miner.
3. **The algorithm implemented here matches the official reference**
   (pearl-research-labs/pearl, zk-pow crate): job_key from header+config,
   keyed-BLAKE3 commitment, low-rank noise, per-tile XOR "jackpot"
   transcripts, `blake3(transcript, key=sigma) ≤ target`. The merkle-tree
   *proof layout* is a conventional binary tree (see `pearlhash/merkle.py`
   for the note on the reference's BLAKE3-native layout).
4. **Token hygiene.** The GitHub token in the repo instructions must never be
   committed. Keep it in `GH_TOKEN` or a git-ignored `config.json`. The token
   you shared in chat should be **revoked/rotated after testing**.

---

## Install

```bash
pip install -r requirements.txt        # numpy, blake3, requests, torch
```

`torch` is used for the GPU matmul. Without CUDA the miner falls back to
`--backend numpy` automatically.

## 0. One-time setup

```bash
# 1. create a private repo (any name, e.g. pearl-miner)
# 2. export the token (NEVER commit it):
export GH_TOKEN=ghp_xxxx
```

## 1. Run the pool connector (one machine, "master")

```bash
python pool_connector.py --workers 3
# or let it ask:
python pool_connector.py
# "How many PCs are joining this mining session? " -> 3
```

It connects one authenticated worker (`krxYRPV4WQ.0x.rig1`, `rig2`, …) to the
pool, publishes every job to `jobs.txt`, and starts watching `shares.txt`.

Options: `--pool` (default `prl.kryptex.network:7048`; TLS: `stratum+ssl://…:8048`),
`--wallet`, `--repo`, `--poll-shares`, `--profile-m/n/k/rank`.

## 2. Run a miner on each PC

```bash
python pearl_miner.py --rig rig1
python pearl_miner.py --rig rig2     # ...on the next PC
```

Every miner polls `jobs.txt`, mines with PyTorch (GPU), and appends shares to
`shares.txt`. The connector submits them to the pool and clears the file.

**Demo mode** (finds shares quickly):
```bash
python pearl_miner.py --rig rig1 --share-target-bits 248
```

## 3. Offline end-to-end test (no pool, no GitHub)

```bash
python selftest.py --m 256 --n 256 --k 1024 --target-bits 248
```
Runs a mock pool + the connector + the miner on a local file bus, and requires
the mock pool to **verify and accept** a mined share (accepted = the math,
protocol and GitHub semantics are all coherent).

Also runnable manually:
```bash
python mock_pool.py --port 19000 --target-bits 248          # terminal 1
python pool_connector.py --bus local --local-dir /tmp/bus \
       --mock localhost:19000 --workers 2                   # terminal 2
python pearl_miner.py --bus local --local-dir /tmp/bus \
       --rig rig1 --share-target-bits 248                   # terminal 3
```

---

## How PearlHash works (what the miner actually does)

Pearl is a Proof-of-Useful-Work coin: mining **is** matrix multiplication.
Per attempt the miner:

1. Builds int8 signal matrices `A (m×k)` and `B (k×n)` from the job header.
2. Computes the commitment `job_key = blake3(header76 ‖ config52)`,
   `b_seed = blake3(job_key ‖ blake3(Bᵀ, key))`,
   `a_seed = blake3(b_seed ‖ blake3(A, key))`.
3. Adds low-rank noise `E_A = E_AL·E_AR`, `E_B = E_BL·E_BR` (derived from the
   seeds) → `A_noised`, `B_noised`.
4. Computes `C = A_noised · B_noised` on the GPU in `rank`-sized chunks; after
   every chunk the XOR-reduction of each `2×64` pattern tile is rotate-XOR-ed
   (rotl 13) into a 16-word **jackpot** transcript.
5. `jackpot_hash = blake3(transcript64, key=a_seed)`; if its little-endian
   value ≤ `target` → **share found**. The share bundles the merkle proofs of
   the opened A rows / B columns so the pool can verify without the full
   matrices.

## Project layout

```
pool_connector.py      master: pool stratum <-> GitHub bus
pearl_miner.py         worker: GitHub bus <-> PyTorch mining
pearlhash/
  params.py            header/config/pattern serialization + difficulty
  commitment.py        job_key, seeds, jackpot hash
  noise.py             low-rank noise generation (official port)
  mining.py            jackpot search (torch/numpy) + share build
  merkle.py            keyed-BLAKE3 chunked merkle proofs
  verify.py            full share verifier (used by the mock pool)
  stratum.py           stratum client + submit builder
github_bus.py          GitHub Contents API (jobs.txt / shares.txt)
mock_pool.py           local mock pool for tests
selftest.py            end-to-end offline test
```

## License / disclaimer

Educational project. Mining has real electricity costs and network effects;
this code is provided as-is, without warranty, for research and testing.
Use of the pool, wallet and token details is entirely your responsibility —
revoke any token you shared and never commit credentials.
