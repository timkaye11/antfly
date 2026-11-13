import json, tempfile, unittest
from pathlib import Path
from unittest import mock
import inclusive_release_gate as gate

class Recompute(unittest.TestCase):
    def put(self, name, value):
        p=self.root/name; p.write_text(json.dumps(value)); return {"path":str(p),"sha256":gate.sha(p)}
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(); self.root=Path(self.t.name)
        self.identity={"artifact":"a"*64}; self.provenance={"binary_sha256":"b"*64}; self.cell={"model":"gemma4-e2b","codec":"f16","context_tokens":8192}
        self.authorities={name:{"path":"/"+name,"sha256":"c"*64} for name in ("source_freeze","full_cli_receipt","binary","matrix","profile","model_config","request_config","native_identity","model_artifact")}
        campaign={"schema":gate.CAMPAIGN_SCHEMA,"passed":True,"schema_version":2,"semantic_identity":self.identity,"provenance":self.provenance,"cells":[self.cell],"warmups":[],"samples":[]}
        cref=self.put("campaign.json",campaign)
        self.report={"schema":gate.REPORT_SCHEMA,"passed":True,"qualification_credit":False,"performance_scope":"full_request_ingestion_through_first_public_content_and_decode","restore_performance_report_substitute":False,"raw_campaign":cref,"execution_authority":self.authorities,"bounds":{"attention_matching_ttft_ratio_to_fastest_exact_upper_95":1.0},"gates":{"attention_matching_ttft":True}}
        self.rref=self.put("report.json",self.report)
        self.manifest={"schema":gate.MANIFEST_SCHEMA,"children":[{"owner_id":"same-owner","artifact_id":"same-artifact"}]}
        self.mref=self.put("manifest.json",self.manifest)
    def tearDown(self): self.t.cleanup()
    def invoke(self, owners=None, artifacts=None):
        return gate.validate({"ok":True},lambda x:x["ok"],[{"report":self.rref,"child_manifest":self.mref}],[self.cell],self.identity,self.provenance,self.authorities,1,owners,artifacts)
    def test_forged_passed_report_cannot_override_recomputed_failed_ttft_gate(self):
        recomputed=dict(self.report); recomputed["passed"]=False; recomputed["bounds"]={"attention_matching_ttft_ratio_to_fastest_exact_upper_95":1.30}; recomputed["gates"]={"attention_matching_ttft":False}
        with mock.patch.object(gate.child_artifacts,"from_campaign",return_value=self.manifest), mock.patch.object(gate.child_artifacts,"revalidate_manifest",return_value={}), mock.patch.object(gate.full_ingestion_performance_audit,"audit",return_value=recomputed):
            with self.assertRaisesRegex(ValueError,"audit report mismatch"): self.invoke()
    def test_owner_uniqueness_set_is_shared_across_model_calls(self):
        owners=set(); artifacts=set()
        with mock.patch.object(gate.child_artifacts,"from_campaign",return_value=self.manifest), mock.patch.object(gate.child_artifacts,"revalidate_manifest",return_value={}), mock.patch.object(gate.full_ingestion_performance_audit,"audit",return_value=self.report):
            self.invoke(owners,artifacts)
            with self.assertRaisesRegex(ValueError,"reused across cells"): self.invoke(owners,artifacts)

if __name__=="__main__": unittest.main()
