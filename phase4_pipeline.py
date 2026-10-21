"""Frozen balanced policy over unchanged BLIP, OCR and Phase 3 features."""
from dataclasses import replace
from phase3_pipeline import Phase3Pipeline
from phase4_policy import verify_policy, run_path, P3


class Phase4Pipeline(Phase3Pipeline):
    def __init__(self, run_dir):
        root=run_path(run_dir)
        policy=verify_policy(root)
        super().__init__(P3)
        self.policy=policy
        self.base.config=replace(self.base.config,evidence_cache_dir=root/'application-evidence-cache')
        self.config=self.base.config

    def run(self,image,question):
        result=super().run(image,question)
        shown,audio,status,debug,annotated=result
        debug['phase4_policy']=self.policy
        debug['phase4_gate_reason']=debug.get('phase3_gate_reason',debug.get('phase3_error'))
        if debug.get('decision')=='phase3_abstained':debug['decision']='phase4_abstained'
        return shown,audio,status,debug,annotated
