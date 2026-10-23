from PIL import Image
from assistive_tools import quality,targeted,compare
from full_text_ocr import assemble
from full_text_session import FullTextSession
from types import SimpleNamespace
import pytest


def evidence(text,box,confidence=.99,identity=None):
    x,y,z,t=box
    return dict(id=identity or text,text=text,box=box,polygon=[[x,y],[z,y],[z,t],[x,t]],confidence=confidence)


def test_lines_order_uncertainty_and_no_completion():
    rows=[evidence('last',[0,90,100,110]),evidence('first',[0,0,100,20]),evidence('maybe',[0,40,100,60],.6),evidence('missing',[0,65,100,85],.1)]
    text,ids,partial=assemble(rows)
    assert text.index('first')<text.index('maybe')<text.index('last')
    assert '[uncertain: maybe]' in text and '[unreadable text]' in text and 'missing' not in text and partial


def test_empty_and_quality_are_nonblocking():
    assert assemble([])==('',[],True)
    q=quality(Image.new('RGB',(100,100),'black'))
    assert q['warnings'] and not q['blocks_inference']


@pytest.mark.parametrize('label,value,question',[('EXP:','12/2027','What is the expiry date?'),
    ('MFG:','01/2026','What is the manufacturing date?'),('PRICE:','Rs 99.00','What is the price?'),('BRAND:','ACME','What brand is this?')])
def test_explicit_inline_field(label,value,question):
    result=targeted(question,[evidence(label+' '+value,[0,0,200,20])])
    assert result['answer_source']=='ocr' and value in result['answer'] and result['supporting_evidence_ids']


def test_no_unlabeled_expiry_or_low_confidence_or_conflicts():
    q='What is the expiry date?'
    assert targeted(q,[evidence('12/2027',[0,0,100,20])])['answer_source']=='abstention'
    assert targeted(q,[evidence('EXP:12/2027',[0,0,100,20],.8)])['answer_source']=='abstention'
    assert targeted(q,[evidence('EXP:12/2027',[0,0,100,20]),evidence('EXP:01/2028',[0,40,100,60])])['answer_source']=='abstention'


def test_spatial_label_value_no_remote_match():
    q='What is the expiry date?';label=evidence('EXP:',[0,0,40,20]);value=evidence('12/2027',[50,0,130,20])
    assert targeted(q,[label,value])['answer_source']=='ocr'
    assert targeted(q,[label,evidence('12/2027',[400,200,480,220])])['answer_source']=='abstention'


def test_router_only_new_facade_and_honest_limits():
    # No initialization or model loads needed to test deterministic routing.
    s=object.__new__(FullTextSession)
    assert s.classify('What is written in the image?').task=='reading'
    assert s.classify('What is the price?').task=='targeted'
    assert s.classify('What brand is this?').task=='targeted'
    assert s.classify('What color is the price tag?').task=='vqa'
    assert s.classify('Is this expired?').task=='unsupported_field'
    assert s.classify('Help').task=='clarify'
    assert s.classify('Transcribe this page.').task=='reading'
    assert s.classify('Read all the words on this brand label.').task=='reading'
    assert s.classify("Don't read all the text.").task=='clarify'
    assert s.classify('Transcribe this and tell me its color.').task=='clarify'


def test_retake_no_correctness_claim_or_photo_storage():
    a={'image_id':'first','quality':quality(Image.new('RGB',(100,100),'black')),'ocr_lines':0}
    b={'image_id':'second','quality':quality(Image.new('RGB',(500,500),'white')),'ocr_lines':3}
    result=compare(a,b)
    assert result['current_ocr_lines']==3 and 'without improving answer correctness' in result['message']
    assert 'image' not in result
