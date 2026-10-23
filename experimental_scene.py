"""Grounded presentation of existing detector candidates; never alters gates."""
import re


def requested(question):
    return bool(re.search(r'\b(?:describe (?:this|the) (?:image|scene)|detailed (?:description|scene)|objects.*(?:visible|identify)|(?:all|what|which) objects)\b',str(question),re.I))


def describe(question,frozen,evidence):
    if not requested(question) or str(frozen.get('final_answer','')).lower()=='unanswerable' or frozen.get('phase3_error'):return None
    from config import Config
    rows=[r for r in evidence if r.get('kind')=='object' and r.get('confidence',0)>=Config().object_presence_threshold]
    names=list(dict.fromkeys(r['text'] for r in rows))
    if not names:return None
    text='Experimental scene overview. The VQA model answers: '+str(frozen['final_answer'])+'. '
    text+='The object detector also identifies these candidates: '+', '.join(names)+'. '
    text+='This is a partial description; detections may miss or misidentify objects.'
    return {'answer':text,'evidence':rows,'supporting_evidence_ids':[r['id'] for r in rows]}
