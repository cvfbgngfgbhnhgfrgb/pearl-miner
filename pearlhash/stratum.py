"""
Minimal Stratum client for the Kryptex Pearl (PRL) pool.

Observed protocol (prl.kryptex.network:7048, plain TCP / :8048 TLS):

  1. mining.configure  [["pearl/v1"], {}]
  2. mining.subscribe  ["pearl-py/0.1.0"]
  3. mining.authorize  ["<wallet>.<worker>", "x"]
  4. server -> mining.notify params = {
        "header":  <76-byte hex>,
        "height":  <int>,
        "job_id":  "<id>_<suffix>",
        "target":  <64-hex uint256>,
        "cert_version": <int> }
     (optionally pearl.set_mining_params / mining.set_difficulty)

Shares are submitted as object params (array params are rejected with
"Unsupported submit format"), see build_submit_params().

A single reader thread parses every line; request/response is matched by id.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import ssl
import threading
import time

log = logging.getLogger("stratum")


class StratumError(Exception):
    pass


class StratumClient:
    def __init__(
        self,
        host: str,
        port: int,
        worker: str,
        *,
        tls: bool = False,
        connect_timeout: float = 12.0,
        keepalive: float = 30.0,
        on_job=None,
        on_submit_result=None,
    ):
        self.host = host
        self.port = port
        self.worker = worker
        self.tls = tls
        self.connect_timeout = connect_timeout
        self.keepalive = keepalive

        self.on_job = on_job or (lambda job: None)
        self.on_submit_result = on_submit_result or (lambda rid, result, error: None)

        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._req_id = 1000
        self._pending: dict[int, dict] = {}
        self._connected = False

        self.job: dict | None = None
        self.set_mining_params: dict | None = None

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        raw = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        if self.tls:
            ctx = ssl.create_default_context()
            if os.environ.get("PEARL_TLS_INSECURE", "0") == "1":
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            raw = ctx.wrap_socket(raw, server_hostname=self.host)
        self._sock = raw
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self._send({"id": 1, "method": "mining.configure", "params": [["pearl/v1"], {}]})
        self._send({"id": 2, "method": "mining.subscribe", "params": ["pearl-py/0.1.0"]})
        resp = self._request(3, "mining.authorize", [self.worker, "x"], timeout=20)
        if not resp.get("result"):
            raise StratumError(f"authorize rejected: {resp}")
        self._connected = True
        log.info("connected to %s:%s as %s (tls=%s)", self.host, self.port, self.worker, self.tls)

    # ------------------------------------------------------------------ #
    def submit(self, params: dict) -> dict:
        """Send mining.submit with object params; returns the response dict."""
        with self._lock:
            self._req_id += 1
            rid = self._req_id
        holder = {"result": None}
        ev = threading.Event()
        with self._lock:
            self._pending[rid] = {"event": ev, "holder": holder}
        self._send({"id": rid, "method": "mining.submit", "params": params})
        ev.wait(timeout=25)
        with self._lock:
            self._pending.pop(rid, None)
        return holder["result"] or {}

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._connected = False

    # ------------------------------------------------------------------ #
    def _request(self, rid: int, method: str, params, timeout: float = 20) -> dict:
        holder = {"result": None}
        ev = threading.Event()
        with self._lock:
            self._pending[rid] = {"event": ev, "holder": holder}
        self._send({"id": rid, "method": method, "params": params})
        ev.wait(timeout=timeout)
        with self._lock:
            self._pending.pop(rid, None)
        if holder["result"] is None:
            raise StratumError(f"no response to {method} (id {rid})")
        return holder["result"]

    def _send(self, obj: dict) -> None:
        if not self._sock:
            raise StratumError("not connected")
        with self._lock:
            self._sock.sendall(json.dumps(obj).encode() + b"\n")

    # ------------------------------------------------------------------ #
    def _read_loop(self) -> None:
        buf = b""
        last_keepalive = time.time()
        while not self._stop.is_set():
            try:
                self._sock.settimeout(1.0)
                d = self._sock.recv(65536)
            except socket.timeout:
                d = b""
            except OSError:
                break
            if d:
                buf += d
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    log.warning("unparseable line: %r", line[:200])
                    continue
                try:
                    self._dispatch(msg)
                except Exception:
                    log.exception("dispatch error")
            if time.time() - last_keepalive > self.keepalive:
                try:
                    self._send({"id": 0, "method": "mining.ping", "params": []})
                except Exception:
                    pass
                last_keepalive = time.time()
        log.info("reader stopped (%s)", self.worker)

    def _dispatch(self, msg: dict) -> None:
        rid = msg.get("id")
        if rid is not None:
            with self._lock:
                holder = self._pending.get(rid)
            if holder:
                holder["holder"]["result"] = msg
                holder["event"].set()
            else:
                log.debug("response for unknown id %s: %s", rid, msg)
            return

        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "mining.notify":
            job = dict(params)
            job["received_ts"] = time.time()
            self.job = job
            log.info("job %s height=%s target=%s", job.get("job_id"), job.get("height"),
                     (job.get("target") or "")[:12])
            self.on_job(job)
        elif method == "mining.set_difficulty":
            log.info("set_difficulty: %s", params)
        elif method == "pearl.set_mining_params":
            self.set_mining_params = dict(params) if isinstance(params, dict) else params
            log.info("pearl.set_mining_params: %s", str(self.set_mining_params)[:200])
        elif method == "pearl.challenge":
            log.info("pearl.challenge received: %s", str(params)[:200])
        elif method == "mining.pong":
            pass
        else:
            log.debug("unknown method %s: %s", method, str(params)[:200])


def build_submit_params(job: dict, share: dict) -> dict:
    """Build the mining.submit params object for a share.

    The Kryptex pool accepts object params; the exact proof encoding is
    pool-specific (krig uses protobuf). This implementation ships a
    documented JSON encoding ("v2-json") that the mock pool verifies
    end-to-end. For the live pool, drop in the pool's own serialization
    here (see README).
    """
    proof = {
        "m": share["m"], "n": share["n"], "k": share["k"], "rank": share["rank"],
        "t_rows": share["t_rows"], "t_cols": share["t_cols"],
        "rows_pattern": share["rows_pattern"], "cols_pattern": share["cols_pattern"],
        "jackpot": share["jackpot"],
        "a_root": share["a_root"], "b_root": share["b_root"],
        "a_tree_root": share["a_tree_root"], "b_tree_root": share["b_tree_root"],
        "a_proof": share["a_proof"], "b_proof": share["b_proof"],
        "jackpot_hash": share["jackpot_hash"],
    }
    return {
        "job_id": share.get("job_id") or job.get("job_id"),
        "type": "v2",
        "header": share["header"],
        "nonce": share["nonce"],
        "sigma": share["sigma"],
        "b_seed": share["b_seed"],
        "target": share["target"],
        "plain_proof": base64.b64encode(json.dumps(proof).encode()).decode(),
    }


def parse_pool_url(url: str, default_port: int = 7048):
    """Parse 'stratum+tcp://host:port', 'host:port' or 'stratum+ssl://...'."""
    tls = False
    host_port = url
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if "ssl" in scheme or "tls" in scheme:
            tls = True
        host_port = rest
    host, _, port_s = host_port.rpartition(":")
    if not host:
        host, port_s = host_port, ""
    port = int(port_s) if port_s else default_port
    if tls and port == default_port and url.startswith("stratum+ssl"):
        port = 8048
    return host, port, tls
