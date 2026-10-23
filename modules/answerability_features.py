"""Fixed inference-only feature allowlist shared by training and application."""
import math
import re

SCHEMA = 'answerability-numeric-v1'
INTENTS = ('general', 'reading', 'price', 'brand', 'date')
STATUSES = ('disabled', 'empty', 'complete', 'partial', 'unavailable', 'unknown')
DECISIONS = ('ocr_supported', 'blip_retained')


def image_quality(image):
    import numpy as np
    from PIL import ImageOps
    image = ImageOps.exif_transpose(image).convert('RGB')
    width, height = image.size
    small = image.convert('L')
    small.thumbnail((256, 256))
    a = np.asarray(small, dtype=np.float32)
    lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4*a[1:-1, 1:-1]
    return {'brightness':float(a.mean()/255), 'contrast':float(a.std()/255),
            'dark_pixels':float((a<=5).mean()), 'light_pixels':float((a>=250).mean()),
            'blur_log_variance':float(math.log1p(float(lap.var()))) if lap.size else 0.,
            'log_width':math.log1p(width),'log_height':math.log1p(height),'aspect_ratio':width/height}


def features(record, quality, normalize=lambda s: ' '.join(s.lower().split())):
    """Never iterate input dictionary fields; ignore annotations and scores."""
    question = record['question']
    from modules.reliability import question_check
    baseline = record['baseline_answer']
    candidate = record['final_answer']
    words = re.findall(r'\w+', question.lower())
    ocr = [e for e in record.get('ocr_evidence', []) if e.get('kind') == 'ocr']
    selected = [e for e in record.get('selected_evidence', []) if e.get('kind') == 'ocr']
    scores = [float(e['confidence']) for e in ocr]
    candidates = record.get('candidate_trace', {}).get('candidates', [])
    out = {key:float(quality[key]) for key in ('brightness','contrast','dark_pixels','light_pixels','blur_log_variance','log_width','log_height','aspect_ratio')}
    out.update(question_words=len(words),question_characters=len(question),answer_words=len(candidate.split()),answer_characters=len(candidate),
               answer_has_digit=float(any(c.isdigit() for c in candidate)),baseline_abstained=float(normalize(baseline)=='unanswerable'),
               candidate_abstained=float(not candidate.strip() or normalize(candidate)=='unanswerable'),candidate_matches_blip=float(normalize(candidate)==normalize(baseline)),
               source_ocr=float(record.get('answer_source')=='ocr'),ocr_fragments=len(ocr),selected_fragments=len(selected),
               high_confidence_fragments=sum(s>=.9 for s in scores),ocr_score_min=min(scores,default=0),
               ocr_score_max=max(scores,default=0),ocr_score_mean=sum(scores)/len(scores) if scores else 0,
               qualifying_candidates=len(candidates),distinct_candidates=len({normalize(c['text']) for c in candidates}),
               supporting_fragments=len(record.get('supporting_evidence_ids',[])),incomplete=float(bool(record.get('incomplete',False))))
    for intent in INTENTS:
        out['intent_'+intent]=float(record.get('ocr_question_intent','general')==intent)
    for component in ('ocr','object'):
        status=record.get('extraction_status',{}).get(component,{}).get('status','unknown')
        for value in STATUSES: out[component+'_status_'+value]=float(status==value)
    for decision in DECISIONS: out['decision_'+decision]=float(record.get('decision')==decision)
    for cue in ('what','how','is','can','please'):
        out['question_cue_'+cue]=float(cue in words)
    out={key:float(value) for key,value in out.items()}
    area=max(1.,math.expm1(quality['log_width'])*math.expm1(quality['log_height']))
    out['selected_ocr_area_ratio']=min(1.,sum(max(0.,(e['box'][2]-e['box'][0])*(e['box'][3]-e['box'][1])) for e in selected)/area)
    out['has_ocr']=float(bool(ocr))
    out['candidate_ambiguity']=float(out['distinct_candidates']>1)
    out['decision_other']=float(record.get('decision') not in DECISIONS)
    out['legacy_question_abstain_hint']=float(question_check(question).decision=='abstain')
    normalized_candidate=normalize(candidate)
    strong_text=[normalize(e['text']) for e in selected if e['confidence']>=.9]
    strong_phrases=[normalize(c['text']) for c in candidates if c.get('confidence',0)>=.9]
    candidate_tokens=set(normalized_candidate.split())
    evidence_tokens={token for text in strong_text for token in text.split()}
    out['candidate_matches_selected_ocr']=float(bool(normalized_candidate) and normalized_candidate in strong_text)
    out['candidate_matches_qualifying_phrase']=float(bool(normalized_candidate) and normalized_candidate in strong_phrases)
    out['candidate_selected_ocr_token_overlap']=len(candidate_tokens & evidence_tokens)/len(candidate_tokens) if candidate_tokens else 0.
    if not all(math.isfinite(value) for value in out.values()):
        raise ValueError('Non-finite inference feature')
    return out
