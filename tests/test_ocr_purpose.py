"""Synthetic OCR content evidence; no models or evaluation data."""
import copy
import pytest
from PIL import Image
from ocr_purpose import describe,purpose_question
from test_reading_relevance import row
from test_image_conversation import fake_base,turn
from image_conversation import prepare
from modules.evidence_store import image_identity


def cooking():
    return [row('Brown ground beef',0,0),row('Stir in hot water,',0,50),row('milk,Sauce Mix and',0,100),row('uncooked Pasta.',0,150),row('Reduce heat.',0,200)]


@pytest.mark.parametrize('q',['What is the image about?','What is this picture about?','What is this document about?'])
def test_purpose_questions(q):assert purpose_question(q)


@pytest.mark.parametrize('q',['What is in the image?','What color is the box?','Describe the scene','How many people?','What is written?'])
def test_other_routes_not_purpose(q):assert not purpose_question(q)


def test_instruction_content_is_extractive_and_not_object_recognition():
    raw={'evidence':cooking()};before=copy.deepcopy(raw);r=describe(raw)
    assert r['used'] and 'directions' in r['answer'] and 'ground beef' in r['answer'] and 'Pasta' in r['answer']
    assert 'not a verified visual scene' in r['answer'] and raw==before
    assert all(x['text'] in r['answer'] for x in r['selected_evidence'])


def test_documents_and_screenshots_do_not_invent_topics_or_active_site():
    rows=[row('This document describes a local event.',0,200),row('Visitors arrive at the main entrance.',0,250),row('The opening time is printed below.',0,300)]
    r=describe({'evidence':rows});assert r['used'] and 'written information' in r['answer']
    rows+=[row('Safari File Edit View History',0,0),row('https://example.com/story.png',0,50)]
    r=describe({'evidence':rows});assert r['used'] and 'browser or interface text' in r['answer'] and 'active website' not in r['answer']


@pytest.mark.parametrize('rows',[[],[row('ACME',0,0)], [row('123 456 789',0,y) for y in (0,40,80)],
    [row('Brown ground beef',0,y,score=.7) for y in (0,40,80)],
    [row('zzzzzz aaaaaa nonsense words',0,y) for y in (0,40,80)],
    [row('asdf qwer zxcv hjkl',0,y) for y in (0,40,80)], [{}]])
def test_unsupported_incidental_or_unreadable_evidence_rejected(rows):assert not describe({'evidence':rows})['used']


def setup(tmp_path):
    base,calls=fake_base(tmp_path,cooking());old=base.reader.read
    def read(image,angle):
        response=old(image,angle);response['result']['image_id']=image_identity(image);return response
    base.reader.read=read
    image=Image.new('RGB',(800,400),'white');return base,calls,image,prepare(None,image)


def test_cached_fallback_preserves_frozen_result_no_new_ocr(tmp_path):
    base,calls,image,state=setup(tmp_path)
    first=turn(base,state,image,'What is the image about?')
    assert first[3]['final_answer']=='unanswerable' and calls==['blip']
    turn(base,state,image,'Read the cooking instructions')
    result=turn(base,state,image,'What is the image about?')
    assert result[3]['answer_source']=='ocr_content_description' and calls==['blip','ocr']
    assert result[3]['frozen_vqa_result']['final_answer']=='unanswerable'
    assert result[3]['baseline_answer']=='unanswerable' and result[3]['purpose_fallback']['cache_only']
    assert result[3]['assistant_trace']['visual_cache_hit']
    assert result[6]['purpose_fallback_ocr_seconds']==0.
    continued=turn(base,state,image,'Continue');assert 'no additional' in continued[0] and calls==['blip','ocr']


def test_blip_success_and_errors_never_overridden(tmp_path):
    base,calls,image,state=setup(tmp_path);turn(base,state,image,'Read cooking steps')
    def success(*args):return ('A box',None,'Complete',{'baseline_answer':'box','final_answer':'box','answer_source':'blip'},None,'',{})
    base.legacy.request=success
    result=turn(base,state,image,'What is this picture about?')
    assert result[0]=='A box' and 'purpose_fallback' not in result[3]
    state['conversation']['visual'].clear()
    def error(*args):return ('',None,'Error',{'final_answer':'unanswerable','phase3_error':'worker failed'},None,'',{})
    base.legacy.request=error
    assert 'purpose_fallback' not in turn(base,state,image,'What is this picture about?')[3]


def test_wrong_image_or_engine_cache_not_reused(tmp_path):
    base,calls,image,state=setup(tmp_path);turn(base,state,image,'Read cooking steps')
    base.engine_choice='EasyOCR'
    result=turn(base,state,image,'What is the image about?');assert result[3]['final_answer']=='unanswerable'
    base.engine_choice='Automatic'
    other=Image.new('RGB',(800,400),'blue');new=prepare(state,other)
    assert turn(base,new,other,'What is the image about?')[3]['final_answer']=='unanswerable'


def test_multiple_coherent_documents_are_not_guessed():
    rows=[row('This is one detailed document about trees.',x,y) for x in (0,1000) for y in (0,40,80)]
    assert describe({'evidence':rows})['reason']=='ambiguous_coherent_regions'


def test_fallback_error_keeps_abstention_with_diagnostic(tmp_path,monkeypatch,caplog):
    base,calls,image,state=setup(tmp_path)
    def broken(*a):raise RuntimeError('private test content')
    monkeypatch.setattr(base,'cached_reading_evidence',broken)
    result=turn(base,state,image,'What is the image about?')
    assert result[3]['final_answer']=='unanswerable' and result[3]['purpose_fallback']['reason']=='fallback_processing_error'
    assert 'stage=ocr-purpose-fallback' in caplog.text and 'private test content' not in caplog.text
