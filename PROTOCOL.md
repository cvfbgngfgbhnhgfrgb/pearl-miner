# Pearl pool protocol notes (observed live, 2026-08-10)

Reverse-engineered from the live pool (`prl.kryptex.network:7048`) and from
strings analysis of the released `krig-miner` (.NET) and `alpha-miner` (C++)
binaries, plus the official `pearl-research-labs/pearl` source.

## Endpoint

| Region | Address |
|---|---|
| Global | `stratum+tcp://prl.kryptex.network:7048` (plain TCP) |
| SSL | `stratum+ssl://prl.kryptex.network:8048` |

Worker name: `WALLET.WORKER` (e.g. `krxYRPV4WQ.0x.rig1`) — authorize with
password `"x"`.

## Handshake (JSON lines, `\n`-terminated)

```
-> {"id":1,"method":"mining.configure","params":[["pearl/v1"],{}]}
-> {"id":2,"method":"mining.subscribe","params":["pearl-py/0.1.0"]}
-> {"id":3,"method":"mining.authorize","params":["krxYRPV4WQ.0x.rig1","x"]}
<- {"id":3,"result":true,"error":null}
```

The pool sends no reply to configure/subscribe. After a successful authorize it
pushes jobs:

```
{"id":null,"method":"mining.notify","params":{
   "header":  "00004020…c50018",            # 76-byte hex (version|prev|merkle|time|nbits)
   "height":  98095,
   "job_id":  "4d64167a_2097152",            # 8-hex + "_" + suffix
   "target":  "00000000000007ff…ff",         # 64-hex uint256 share target
   "cert_version": 2}}
```

## Shares

`mining.submit` with **object params** (arrays → `error [20,"Unsupported submit
format"]`). Known fields from krig: `job_id`, `type` (`"v2"`), `plain_proof`
(base64), plus `sigma` (a_seed), `b_seed`, header/nonce. Missing/unknown job →
`error [21,"Job not found"]`.

This repo ships a documented `v2-json` proof encoding (see
`pearlhash/stratum.py: build_submit_params`) verified end-to-end by
`mock_pool.py`. Kryptex's own krig-miner encodes the proof as a protobuf
(`pearlpool.mining.v2.JobAssignment`, `ToProtoMerkleProof`) — the submit
builder is the single place to plug that in for production.

## PearlHash algorithm (reference: zk-pow crate)

- `job_key = blake3(header76 ‖ config52)`
- `hash_a = blake3(pad1024(A_rowmajor), key=job_key)`,
  `hash_b = blake3(pad1024(Bᵀ), key=job_key)`
- `b_seed = blake3(job_key ‖ hash_b)`, `a_seed = blake3(b_seed ‖ hash_a)`
- noise: `E_A = E_AL·E_AR`, `E_B = E_BL·E_BR` (uniform + permutation draws
  from keyed BLAKE3; rank r ∈ {32..1024}, default 128)
- tile: `rows_pattern.size() × cols_pattern.size()` = 2 × 64 for the default
  profile; C accumulated in `rank` chunks; per chunk
  `jackpot[tid] = rotl13(jackpot[tid]) ^ xor(tile)`, `tid = chunk % 16`
- `jackpot_hash = blake3(transcript64, key=a_seed)`; share iff
  `int.from_bytes(jackpot_hash, "little") <= target`
- default profile: m=n=2048 (miner flag), k=1024, rank=128,
  rows=[0,8], cols=[0,1,8,9,…,248,249]; `DAF = 128 · k`
