"""Opt-in wrapper: preserve existing application and frozen Phase 2 generation."""
from dataclasses import replace
import json
import os

from config import Config
from evaluate_vqa import OfficialScorer
from modules.answerability_features import features,image_quality
from modules.answerability_gate import Gate,decide
from modules.question_intent import classify
from phase3_common import CHECKPOINT,MODEL_CACHE,DEV_INPUT,safe_run
from utils.image_utils import normalize_image


class Phase3Pipeline:
    def __init__(self,run_dir):
        from pipeline import Pipeline
        root=safe_run(run_dir)
        self.gate=Gate(root/'models')
        self.policy=json.loads((root/'selected-policy.json').read_text())
        if self.policy['ocr_threshold']!=.9 or not self.policy['development_selected']:
            raise ValueError('Expected frozen OCR and development-selected gate policy')
        if self.policy['threshold_reliable'] is not None and self.gate.models.get('reliability') is None:
            raise ValueError('Selected reliability head unavailable; no silent fallback')
        self.scorer=OfficialScorer(json.loads(DEV_INPUT.read_text())['official_rules'])
        os.environ['YOLO_CONFIG_DIR']=str(root/'runtime/ultralytics')
        cfg=replace(Config(),vqa_mode='v2',improved_blip_dir=CHECKPOINT,device='cuda',ocr_device='cpu',object_device='cpu',
                    cache_dir=MODEL_CACHE,yolo_model=str(MODEL_CACHE/'yolov8n.pt'),evidence_cache_dir=root/'application-evidence-cache',ocr_min_score=.9)
        self.base=Pipeline(cfg)

    def run(self,image,question):
        shown,audio,status,debug,annotated=self.base.run(image,question,speak=False,vqa_mode='v2',evidence_aware=True)
        if debug.get('baseline_answer') is None:
            debug['phase3_error']='Base inference failed; learned gate was not applied.'
            return shown,audio,status,debug,annotated
        intent=classify(question)
        statuses={('object' if k=='objects' else k):v for k,v in debug['extraction_status'].items()}
        statuses.setdefault('object',{'status':'disabled'});statuses.setdefault('ocr',{'status':'disabled'})
        record={**debug,'question':question,'ocr_question_intent':intent.kind,
                'ocr_evidence':[e for e in debug['evidence'] if e.get('kind')=='ocr'],
                'extraction_status':statuses,'incomplete':any(v.get('status') in ('partial','unavailable') for v in statuses.values())}
        quality=image_quality(normalize_image(image,thumbnail=False))
        x=features(record,quality,self.scorer.normalize)
        pa,pr=self.gate.predict(x,record['final_answer'])
        decision=decide(record['final_answer'],pa,pr,self.policy['threshold_answerable'],self.policy['threshold_reliable'])
        debug.update(phase2_final_answer=record['final_answer'],phase2_decision_reason=debug['decision_reason'],
                     phase2_decision=debug['decision'],phase2_supporting_evidence_ids=list(debug['supporting_evidence_ids']),
                     phase2_answer_source=debug['answer_source'],phase2_grounded=debug['grounded'],
                     final_answer=decision['answer'],phase3_gate_reason=decision['reason'],
                     answerability_probability=pa,candidate_full_agreement_probability=pr,
                     phase3_policy=self.policy,probabilities_do_not_prove_visual_support=True)
        debug['answerability']={'assessed':True,'probability':pa,'target':'VizWiz answerable label',
                               'passes_selected_threshold':None if self.policy['threshold_answerable'] is None else pa>=self.policy['threshold_answerable']}
        debug['candidate_reliability']={'assessed':pr is not None,'probability':pr,'target':'full reference agreement for a non-abstaining candidate'}
        if decision['abstained']:
            debug['grounded']=False;debug['answer_source']='abstention'
            debug['decision']='phase3_abstained';debug['supporting_evidence_ids']=[]
            shown='I do not have enough reliable information to answer. Please try a clearer photo or another question.'
        else: shown=decision['answer']
        return shown,None,status,debug,annotated
