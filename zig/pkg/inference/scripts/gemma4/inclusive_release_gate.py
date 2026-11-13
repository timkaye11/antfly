#!/usr/bin/env python3
"""Fail-closed composition of canonical restore and full-ingestion gates."""
import hashlib, json, re
from pathlib import Path
import child_artifacts
import full_ingestion_performance_audit

REPORT_SCHEMA = "antfly.compaction-full-ingestion-performance.v1"
CAMPAIGN_SCHEMA = "antfly.attention-matching-full-ttft-campaign.v1"
MANIFEST_SCHEMA = "antfly.full-ingestion-child-artifact-manifest.v1"
REQUIRED_CELLS = 12
HEX = re.compile(r"[0-9a-f]{64}")

def need(v, message):
    if not v: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def load_ref(ref, label):
    need(type(ref) is dict and set(ref)=={"path","sha256"}, label+" descriptor")
    p=Path(ref["path"])
    need(p.is_absolute() and p.is_file() and not p.is_symlink() and HEX.fullmatch(ref["sha256"]) and sha(p)==ref["sha256"], label+" drift")
    return json.loads(p.read_bytes())

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",",":"))

def validate(canonical_restore_report, canonical_restore_validator, inclusive, expected_cells,
             expected_identity, expected_provenance, expected_authorities, required_cells=REQUIRED_CELLS,
             global_owner_ids=None, global_artifact_ids=None):
    # Existing canonical restore semantics and numeric gates remain authoritative.
    canonical_result=canonical_restore_validator(canonical_restore_report)
    need(canonical_result is not False, "canonical restore gate failed")
    need(type(required_cells) is int and required_cells > 0 and type(expected_cells) is list and
         len(expected_cells)==required_cells and len({canonical(x) for x in expected_cells})==required_cells, "expected cell authority")
    need(type(inclusive) is list and len(inclusive)==required_cells, "requires exact inclusive cell count")
    expected={canonical(x) for x in expected_cells}; seen=set()
    owners = set() if global_owner_ids is None else global_owner_ids
    artifacts = set() if global_artifact_ids is None else global_artifact_ids
    for item in inclusive:
        need(type(item) is dict and set(item)=={"report","child_manifest"}, "inclusive entry")
        report=load_ref(item["report"], "inclusive report")
        need(report.get("schema")==REPORT_SCHEMA and report.get("passed") is True and
             report.get("qualification_credit") is False and
             report.get("performance_scope")=="full_request_ingestion_through_first_public_content_and_decode" and
             report.get("restore_performance_report_substitute") is False, "restore-only substitution")
        campaign=load_ref(report.get("raw_campaign"), "raw inclusive campaign")
        need(campaign.get("schema")==CAMPAIGN_SCHEMA and campaign.get("passed") is True, "inclusive campaign")
        header={key:campaign.get(key) for key in ("schema_version","semantic_identity","provenance","cells")}
        need(type(header) is dict and header.get("schema_version")==2 and
             header.get("semantic_identity")==expected_identity and
             header.get("provenance")==expected_provenance, "mixed source or native identity")
        cells=header.get("cells")
        need(type(cells) is list and len(cells)==1, "inclusive report must bind one cell")
        key=canonical(cells[0]); need(key in expected and key not in seen, "missing or duplicate inclusive cell"); seen.add(key)
        authority=report.get("execution_authority")
        need(type(authority) is dict, "execution authority")
        for name, expected_ref in expected_authorities.items():
            need(authority.get(name)==expected_ref, "mixed "+name)
        recomputed = full_ingestion_performance_audit.audit(campaign, report["raw_campaign"], {
            "schema": "antfly.current-full-ingestion-server-launch.v1", "passed": True,
            "qualification_claim": False, "performance_scope": "full_request_ingestion",
            "performance_header": header, **authority})
        need(report == recomputed, "full-ingestion audit report mismatch")
        manifest=load_ref(item["child_manifest"], "child manifest")
        need(manifest.get("schema")==MANIFEST_SCHEMA and manifest==child_artifacts.from_campaign(campaign), "child manifest mismatch")
        child_artifacts.revalidate_manifest(item["child_manifest"]["path"])
        for child in manifest["children"]:
            need(child["owner_id"] not in owners and child["artifact_id"] not in artifacts, "owner or artifact reused across cells")
            owners.add(child["owner_id"]); artifacts.add(child["artifact_id"])
    need(seen==expected, "missing inclusive cell")
    return {"canonical_restore_gate": canonical_result, "full_ingestion_cells": required_cells,
            "full_ingestion_gate": True, "semantics": "canonical_restore_plus_full_ingestion_extra_gate"}
