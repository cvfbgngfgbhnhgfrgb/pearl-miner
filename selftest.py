#!/usr/bin/env python3
"""
selftest.py — full offline end-to-end test:

    mock pool  <-  pool_connector  ->  jobs.txt (local bus)
                                          |
                                      pearl_miner (mines)
                                          |
    mock pool  <-  pool_connector  <-  shares.txt (local bus)

The mock pool fully verifies each share with the local verifier, so an
"accepted" share proves the whole pipeline (protocol + math) is coherent.

Usage:  python selftest.py [--target-bits 250] [--m 1024] [--n 1024]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from mock_pool import MockPool

log = logging.getLogger("selftest")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-bits", type=int, default=250)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    workdir = tempfile.mkdtemp(prefix="pearl_selftest_")
    busdir = f"{workdir}/bus"

    # ---- mock pool -----------------------------------------------------
    pool = MockPool(0, args.target_bits)          # port 0 -> ephemeral
    pool.port = 0
    # rebind with an ephemeral port
    pool._srv.close()
    pool._srv = __import__("socket").socket(__import__("socket").AF_INET,
                                            __import__("socket").SOCK_STREAM)
    pool._srv.setsockopt(__import__("socket").SOL_SOCKET, __import__("socket").SO_REUSEADDR, 1)
    pool._srv.bind(("0.0.0.0", 0))
    pool._srv.listen(8)
    port = pool._srv.getsockname()[1]
    pool.port = port
    log.info("mock pool on port %d", port)

    t = threading.Thread(target=pool.run, daemon=True)
    t.start()
    time.sleep(0.3)

    # ---- pool connector (subprocess) -----------------------------------
    conn_cmd = [
        sys.executable, "pool_connector.py",
        "--bus", "local", "--local-dir", busdir,
        "--mock", f"localhost:{port}",
        "--workers", str(args.workers),
        "--profile-m", str(args.m), "--profile-n", str(args.n),
        "--profile-k", str(args.k), "--profile-rank", "128",
        "--poll-shares", "1.0",
    ]
    log.info("starting pool connector: %s", " ".join(conn_cmd))
    conn = subprocess.Popen(conn_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)

    # ---- pearl miner (subprocess) --------------------------------------
    miner_cmd = [
        sys.executable, "pearl_miner.py",
        "--bus", "local", "--local-dir", busdir,
        "--rig", "selftest-rig",
        "--m", str(args.m), "--n", str(args.n), "--k", str(args.k),
        "--rank", "128",
        "--share-target-bits", str(args.target_bits),
        "--poll-jobs", "1.0",
    ]
    log.info("starting pearl miner: %s", " ".join(miner_cmd))
    miner = subprocess.Popen(miner_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)

    # ---- wait for an accepted share --------------------------------------
    deadline = time.time() + args.timeout
    ok = False
    try:
        while time.time() < deadline:
            if pool.accepted > 0:
                ok = True
                break
            if conn.poll() is not None or miner.poll() is not None:
                log.error("a component exited early; dumping logs")
                break
            time.sleep(0.5)
    finally:
        for proc in (conn, miner):
            try:
                out, _ = proc.communicate(timeout=3)
                if out:
                    log.info("---- %s output ----\n%s", proc.args[1], out[-4000:])
            except Exception:
                proc.kill()

    log.info("mock pool stats: accepted=%d rejected=%d reasons=%s",
             pool.accepted, pool.rejected, pool.reject_reasons)
    if ok:
        print("\n✅ SELFTEST PASSED: full pipeline produced an accepted share.")
        print(f"   (accepted={pool.accepted}, rejected={pool.rejected})")
        shutil.rmtree(workdir, ignore_errors=True)
        return 0
    print("\n❌ SELFTEST FAILED: no accepted share within timeout.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
