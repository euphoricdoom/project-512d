"""Loop / project-512d export hook for Sun_Tan.

This command creates a Sun_Tan-compatible bridge packet for a Loop artifact
without requiring project-512d to depend on the Sun_Tan package.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_SALT = "suntan-local-v0"


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sign(payload: str, salt: str = DEFAULT_LOCAL_SALT) -> str:
    return sha256(f"{salt}:{payload}".encode("utf-8")).hexdigest()


def build_suntan_packet(
    artifact: str | Path,
    out: str | Path,
    lineage: list[str] | None = None,
    policy: str = "policy_v1",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bridge_version": "0.1",
        "source_system": "project-512d",
        "target_system": ".Neon",
        "artifact_hash": f"sha256:{sha256_file(artifact)}",
        "created_at": datetime.now(UTC).isoformat(),
        "policy": policy,
        "lineage": lineage or [],
        "pulse_hash": None,
    }

    payload["payload_hash"] = f"sha256:{sha256_text(canonical_json(payload))}"
    payload["signature"] = sign(canonical_json(payload))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loop export-suntan")
    parser.add_argument("artifact")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lineage", action="append", default=[])
    parser.add_argument("--policy", default="policy_v1")
    args = parser.parse_args(argv)

    build_suntan_packet(
        artifact=args.artifact,
        out=args.out,
        lineage=args.lineage,
        policy=args.policy,
    )
    print(f"Sun_Tan packet written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
