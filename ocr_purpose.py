"""Experimental content descriptions from cached OCR, never visual recognition."""
import re
import time
import math
from reading_sequences import accepted,reading_order
from reading_relevance import group_regions

VERSION='cached-ocr-purpose-v1'


def purpose_question(question):
    q=' '.join(str(question).lower().strip(' .?!').split())
    return bool(re.fullmatch(r'(?:please )?(?:what is (?:this |the )?(?:image|picture|photo|document|label|screenshot|page) about|what is the purpose of (?:this |the )?(?:image|document|label|page)|what information (?:is shown|is present) in (?:this |the )?image)',q))


def describe(details):
    """Layout/coherence gates are heuristics, not correctness probabilities.

    Category labels require structural evidence. All descriptive detail is an
    extractive quotation; no ingredients, objects, brands or topics are supplied
    from a template, external knowledge or the previous assistant answer.
    """
    start=time.perf_counter();rows=[]
    for r in details.get('evidence',[]):
        if not isinstance(r,dict):continue
        b=r.get('corrected_box',r.get('box'))
        if (not isinstance(r.get('id'),str) or not isinstance(r.get('text'),str) or
            r.get('source',{}).get('engine') not in ('RapidOCR','EasyOCR') or
            not isinstance(r.get('confidence'),(int,float)) or not math.isfinite(r['confidence']) or
            not isinstance(b,(list,tuple)) or len(b)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in b)):continue
        rows.append(r)
    groups=group_regions(rows);by_id={r['id']:r for r in rows}
    candidates=[]
    for g in groups:
        valid=[by_id[i] for i in g['accepted_ids'] if i in by_id and accepted(by_id[i])]
        words=[w for r in valid for w in re.findall(r"[A-Za-z][A-Za-z'-]+",r['text'])]
        if len(valid)<3 or len(words)<12 or sum(len(r['text'].split())>=3 for r in valid)<2:continue
        if len(valid)/max(1,len(g['evidence_ids']))<.75:continue
        if any(re.search(r'([a-zA-Z])\1{4,}',r['text']) for r in valid):continue
        if g['kind'] not in ('instruction_sequence','main_passage'):continue
        if g['kind']!='instruction_sequence' and len(set(w.lower() for w in words)&{'the','and','in','of','to','is','for','with','a','an','at','from','this','are','on'})<2:continue
        candidates.append((g,valid))
    instructions=[c for c in candidates if c[0]['kind']=='instruction_sequence' and c[0]['signals'].get('command_count',0)>=2]
    if instructions:candidates=instructions
    if len(candidates)!=1:
        return {'used':False,'reason':'ambiguous_coherent_regions' if candidates else 'insufficient_coherent_text','version':VERSION}
    group,valid=candidates[0];ordered=reading_order(valid)
    # Bound speech without guessing topic keywords. Quote complete source lines.
    chosen=[];characters=0
    for row in ordered:
        if len(row['text'])>160:continue
        if chosen and characters+len(row['text'])>220:break
        chosen.append(row);characters+=len(row['text'])
        if len(chosen)>=5:break
    if len(chosen)<2:return {'used':False,'reason':'no_concise_supported_excerpt','version':VERSION}
    interface=sum(g['kind'] in ('browser_toolbar','website_metadata') for g in groups)>=2
    if group['kind']=='instruction_sequence':category='cooking or preparation directions'
    elif interface:category='a written passage alongside browser or interface text'
    else:category='a passage of written information'
    answer='The recognized text describes '+category+'. It includes: '+'; '.join('“'+r['text']+'”' for r in chosen)+'.'
    answer+=' This describes the printed content, not a verified visual scene. The reading may be incomplete.'
    ids=[r['id'] for r in chosen]
    return {'used':True,'answer':answer,'reason':'unique_coherent_cached_text_region','version':VERSION,
            'description_kind':'ocr_grounded_printed_content','selected_evidence':chosen,'supporting_evidence_ids':ids,
            'selected_region_ids':[group['id']],'regions':groups,'category':category,
            'passage_candidates':[dict(id=g['id'],box=g['box'],accepted_ids=g['accepted_ids'],preview=g['text'][:120]) for g in groups if g['accepted_ids']],
            'composition_seconds':time.perf_counter()-start}
