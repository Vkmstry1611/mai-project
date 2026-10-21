from PIL import Image
from phase3_pipeline import Phase3Pipeline


def test_gate_is_after_phase2_and_candidate_evidence_preserved():
    class Base:
        def run(self,image,question,**kwargs):
            assert kwargs['evidence_aware'] and kwargs['vqa_mode']=='v2' and not kwargs['speak']
            return 'MILK',None,'Complete',{'baseline_answer':'water','final_answer':'MILK','decision_reason':'OCR supported',
                'answer_source':'ocr','grounded':True,'decision':'ocr_supported','extraction_status':{},'candidate_trace':{},
                'evidence':[],'selected_evidence':[],'supporting_evidence_ids':['ocr-stable']},image
    class Gate:
        def predict(self,x,candidate):
            assert candidate=='MILK' and x['answer_characters']==4
            return .4,.3
    class Scorer:
        def normalize(self,text):return text.lower()
    wrapper=Phase3Pipeline.__new__(Phase3Pipeline);wrapper.base=Base();wrapper.gate=Gate();wrapper.scorer=Scorer()
    wrapper.policy={'threshold_answerable':.5,'threshold_reliable':.5}
    shown,_,_,debug,_=wrapper.run(Image.new('RGB',(24,24)),'Read this label')
    assert debug['baseline_answer']=='water' and debug['phase2_final_answer']=='MILK'
    assert debug['final_answer']=='unanswerable' and debug['phase2_supporting_evidence_ids']==['ocr-stable']
    assert debug['supporting_evidence_ids']==[] and debug['decision']=='phase3_abstained'
    assert debug['phase2_decision_reason']=='OCR supported' and debug['phase2_answer_source']=='ocr'
    assert debug['grounded'] is False and debug['probabilities_do_not_prove_visual_support']
