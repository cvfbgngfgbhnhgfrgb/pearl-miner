#!/usr/bin/env python3
"""
mock_pool.py — a local mock of the Pearl stratum pool for end-to-end tests.

Speaks the same JSON-lines stratum as prl.kryptex.network:7048:

  client -> mining.configure / mining.subscribe / mining.authorize
  server -> mining.notify  (header/height/job_id/target/cert_version)
  client -> mining.submit  (object params, "v2-json" encoding)
  server -> verifies the share locally (pearlhash.verify) and replies
            {"id": N, "result": true/false, "error": {...}}

Run:  python mock_pool.py --port 19000 --target-bits 250
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import socket
import struct
import threading
import time

from pearlhash.verify import VerifyError, verify_share

log = logging.getLogger("mockpool")


def make_job(height: int, target_bits: int) -> dict:
    version = 0x20400000
    prev = bytes(random.getrandbits(8) for _ in range(32))
    merkle = bytes(random.getrandbits(8) for _ in range(32))
    ts = int(time.time())
    # network nbits chosen so nbits_to_difficulty ~ 2^40 (easy chain)
    nbits = 0x1c00FFFF  # any value is fine; the explicit target drives shares
    header = (struct.pack("<I", version) + prev + merkle
              + struct.pack("<I", ts) + struct.pack("<I", nbits))
    job_id = f"{random.randrange(16**8):08x}_2097152"
    target = f"{1 << target_bits:064x}"
    return {
        "header": header.hex(),
        "height": height,
        "job_id": job_id,
        "target": target,
        "cert_version": 2,
    }


class MockPool:
    def __init__(self, port: int, target_bits: int, job_interval: float = 0.0):
        self.port = port
        self.target_bits = target_bits
        self.job_interval = job_interval
        self.height = 100000
        self.jobs: dict[str, dict] = {}
        self.accepted = 0
        self.rejected = 0
        self.reject_reasons: dict[str, int] = {}
        self._lock = threading.Lock()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("0.0.0.0", port))
        self._srv.listen(8)

    def run(self):
        log.info("mock pool listening on 0.0.0.0:%d target_bits=%d",
                 self.port, self.target_bits)
        stop = threading.Event()

        def broadcaster():
            while not stop.is_set():
                time.sleep(self.job_interval or 30)
                with self._lock:
                    self.height += 1
                # connections pick up new jobs when they ask; simplest is
                # to push on the next accepted connection
        threading.Thread(target=broadcaster, daemon=True).start()

        while not stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

    def _handle(self, conn: socket.socket, addr):
        log.info("client connected from %s", addr)
        buf = b""
        conn.settimeout(2.0)
        try:
            while True:
                try:
                    d = conn.recv(65536)
                except socket.timeout:
                    continue          # idle is fine — keep the connection
                if not d:
                    break             # EOF
                buf += d
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except Exception:
                        continue
                    self._on_message(conn, msg)
        except Exception as e:
            log.debug("conn error: %s", e)
        finally:
            conn.close()

    def _send(self, conn, obj):
        try:
            conn.sendall(json.dumps(obj).encode() + b"\n")
        except OSError:
            pass

    def _on_message(self, conn, msg):
        method = msg.get("method")
        rid = msg.get("id")
        params = msg.get("params") or []

        if method == "mining.configure":
            return
        if method == "mining.subscribe":
            return
        if method == "mining.authorize":
            self._send(conn, {"id": rid, "result": True, "error": None})
            job = make_job(self.height, self.target_bits)
            with self._lock:
                self.jobs[job["job_id"]] = job
            self._send(conn, {"id": None, "method": "mining.notify", "params": job})
            return
        if method == "mining.ping":
            self._send(conn, {"id": rid, "result": "pong", "error": None})
            return
        if method == "mining.submit":
            self._on_submit(conn, rid, params)
            return
        log.debug("unhandled: %s", str(msg)[:160])

    def _on_submit(self, conn, rid, params):
        if not isinstance(params, dict):
            self._send(conn, {"id": rid, "error": {"code": -1, "message": [20, "Unsupported submit format", None]}, "result": None})
            return
        job_id = params.get("job_id")
        with self._lock:
            job = self.jobs.get(job_id)
        if not job:
            self._send(conn, {"id": rid, "error": {"code": -1, "message": [21, "Job not found", None]}, "result": None})
            return
        try:
            import base64
            proof = json.loads(base64.b64decode(params["plain_proof"]))
            share = dict(proof)
            share["header"] = params["header"]
            share["nonce"] = params["nonce"]
            share["sigma"] = params["sigma"]
            share["b_seed"] = params["b_seed"]
            share["target"] = params["target"]
            share["job_id"] = job_id
            result = verify_share(share)
            self.accepted += 1
            self._send(conn, {"id": rid, "result": True, "error": None})
            log.info("*** SHARE ACCEPTED job=%s tile=(%s,%s) jackpot=%s",
                     job_id, share.get("t_rows"), share.get("t_cols"),
                     share.get("jackpot_hash", "")[:16])
        except (VerifyError, KeyError, ValueError, Exception) as e:  # noqa: B014
            self.rejected += 1
            self.reject_reasons[type(e).__name__] = self.reject_reasons.get(type(e).__name__, 0) + 1
            self._send(conn, {"id": rid, "result": False,
                              "error": {"code": -1, "message": [22, f"Share rejected: {e}", None]}})
            log.info("share REJECTED job=%s reason=%s", job_id, e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=19000)
    p.add_argument("--target-bits", type=int, default=250)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.debug else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    MockPool(a.port, a.target_bits).run()
