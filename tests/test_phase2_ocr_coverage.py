"""Coverage completion regressions: fake OCR, no model or dataset downloads."""
import copy
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from visionbridge.complete_phase2_ocr import missing_indices,apply_ocr,reader_rows
from test_phase2_dev import row,captured,item,rules
from replay_phase2_dev import reselect_capture
from evaluate_phase2_dev import replay
from evaluate_vqa import OfficialScorer


def missing(question='Can you tell me what this says, please?'):
    r=row(question=question);c=captured(r,evidence=[])
    c['extraction_status']['ocr']={'status':'disabled'}
    c['ocr_supported_question']=False;c['image_identity']='image-fixture'
    return r,c


def test_exact_missing_selection_excludes_unsupported_and_already_extracted():
    _,c=missing();_,unsupported=missing('Is this red?')
    _,done=missing();done['extraction_status']['ocr']={'status':'empty'}
    assert missing_indices([unsupported,c,done])==[1]


def test_overlay_preserves_baseline_objects_and_original_capture():
    r,c=missing();c['evidence']=[dict(kind='object',id='object-a',text='laptop')]
    before=copy.deepcopy(c);e=dict(item(),image_id='image-fixture')
    result=apply_ocr(c,[e],{'status':'complete','rejected_items':0},2.)
    assert c==before and result['baseline_answer']==c['baseline_answer']
    assert result['evidence'][0]==c['evidence'][0]
    scored=replay(r,reselect_capture(result),OfficialScorer(rules()),.9)
    assert scored['final_answer']=='WELCOME'
    assert scored['answers']==r['answers']
    with pytest.raises(ValueError,match='replace'):apply_ocr(result,[e],{'status':'complete'},1.)


def test_empty_partial_and_failed_extraction_never_invent_evidence():
    r,c=missing()
    empty=apply_ocr(c,[],{'status':'empty'},1.)
    assert replay(r,reselect_capture(empty),OfficialScorer(rules()),.9)['answer_source']=='blip'
    e=dict(item(),image_id='image-fixture')
    partial=apply_ocr(c,[e],{'status':'partial','rejected_items':1},1.)
    assert replay(r,reselect_capture(partial),OfficialScorer(rules()),.9)['answer_source']=='blip'
    with pytest.raises(RuntimeError):apply_ocr(c,[],{'status':'unavailable'},1.)
    with pytest.raises(ValueError,match='identity'):apply_ocr(c,[dict(e,image_id='wrong')],{'status':'complete'},1.)


def test_reader_conversion_matches_existing_ocr_contract():
    def readtext(array):
        assert isinstance(array,np.ndarray) and array.shape==(20,30,3)
        return [([[2,3],[12,3],[12,9],[2,9]],'WELCOME',.98)]
    values=reader_rows(SimpleNamespace(readtext=readtext),Image.new('RGB',(30,20)))
    assert values[0]['box']==[2.,3.,12.,9.]
    assert values[0]['confidence']==.98 and values[0]['text']=='WELCOME'
