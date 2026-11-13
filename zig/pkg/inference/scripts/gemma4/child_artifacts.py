"""Closed child-artifact manifest for full-ingestion campaigns."""
import hashlib, json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def descriptor(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

def validate(value, label):
    if type(value) is not dict or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError(label + " descriptor")
    path = Path(value["path"])
    if (not path.is_absolute() or not path.is_file() or path.is_symlink() or
        type(value["bytes"]) is not int or value["bytes"] < 0 or
        path.stat().st_size != value["bytes"] or sha256(path) != value["sha256"]):
        raise ValueError(label + " drift")
    return path

def from_campaign(campaign):
    children, paths = [], set()
    rows = campaign.get("warmups", []) + campaign.get("samples", [])
    if len(rows) != 35:
        raise ValueError("campaign triplet count")
    for row in rows:
        for arm in ("exact", "packed_exact", "attention_matching"):
            authority = row.get(arm, {}).get("full_ttft_authority", {})
            item = authority.get("artifact_authority")
            path = validate(item, arm + " artifact authority")
            if str(path) in paths:
                raise ValueError("child path reused")
            paths.add(str(path)); children.append({"kind": "request_authority", "arm": arm,
                "owner_id": authority.get("fresh_owner_id"), "artifact_id": authority.get("fresh_artifact_id"), **item})
            checkpoint = authority.get("checkpoint")
            if arm == "attention_matching":
                path = validate(checkpoint, "attention matching checkpoint")
                if str(path) in paths:
                    raise ValueError("child path reused")
                paths.add(str(path)); children.append({"kind": "attention_matching_checkpoint", "arm": arm,
                    "owner_id": authority.get("fresh_owner_id"), "artifact_id": authority.get("fresh_artifact_id"), **checkpoint})
            elif checkpoint is not None:
                raise ValueError("exact arm checkpoint")
    if len(children) != 140 or sum(x["kind"] == "request_authority" for x in children) != 105 or sum(x["kind"] == "attention_matching_checkpoint" for x in children) != 35:
        raise ValueError("child cardinality")
    children.sort(key=lambda x: (x["kind"], x["artifact_id"], x["path"]))
    return {"schema": "antfly.full-ingestion-child-artifact-manifest.v1", "children": children,
            "request_authorities": 105, "attention_matching_checkpoints": 35,
            "warmup_artifacts_retained": True}

def write_exclusive(path, campaign):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(from_campaign(campaign), stream, sort_keys=True, indent=2)
        stream.write("\n"); stream.flush()
    return descriptor(path)

def revalidate_manifest(path):
    manifest = json.loads(Path(path).read_bytes())
    if manifest.get("schema") != "antfly.full-ingestion-child-artifact-manifest.v1" or len(manifest.get("children", [])) != 140:
        raise ValueError("child manifest")
    for index, child in enumerate(manifest["children"]):
        validate({key: child[key] for key in ("path", "sha256", "bytes")}, f"child {index}")
    return descriptor(path)
