# Sun_Tan Export Contract

## Purpose

Allow Loop / `project-512d` to export artifacts into the unified product spine without depending on Sun_Tan internals.

```text
Loop artifact
→ Sun_Tan-compatible bridge packet
→ Sun_Tan verification
→ .Neon origin claim
```

## Command

```bash
python suntan_export.py <artifact> --out <packet.json>
```

Optional lineage:

```bash
python suntan_export.py <artifact> --lineage .N/example-root --out <packet.json>
```

## Packet behavior

The export hook writes a Sun_Tan-compatible bridge packet with:

- `source_system = project-512d`
- `target_system = .Neon`
- `artifact_hash = sha256:<artifact-bytes>`
- `payload_hash = sha256:<canonical-payload>`
- deterministic local signature
- optional lineage references
- default `policy_v1`

## Boundary

Loop owns cognition, training, proof, and model artifacts.

Sun_Tan owns bridge verification and claim export.

`.Neon` owns continuity import, lineage, and long-term traversal.
