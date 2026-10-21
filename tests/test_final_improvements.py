"""Synthetic evidence regressions; manual expectations are never model labels."""
import pytest
from PIL import Image
from conversation_recall import recall
from assistive_tools import targeted,field_intent
from full_text_session import FullTextSession
from image_conversation import prepare
from test_image_conversation import fake_base,turn,row
from test_text_hierarchy import evidence


def test_recall_exact_no_models(tmp_path):
    base,calls=fake_base(tmp_path,evidence());im=Image.new('RGB',(800,600));state=prepare(None,im)
    turn(base,state,im,'Read the first cooking step')
    turn(base,state,im,'What is this image about?')
    before=list(calls)
    r=turn(base,state,im,'Could you repeat my second question?')
    assert r[0]=='Your second question was: What is this image about?'
    assert calls==before and r[3]['answer_source']=='session_history'
    assert 'Brown' in turn(base,state,im,'Could you repeat the previous step?')[0]
    assert calls==before
    assert len({t['id'] for t in state['conversation']['turns']})==4
    assert not prepare(None,im)['conversation']['turns']


@pytest.mark.parametrize('q',['What was my third question?','What did I ask earlier?','Repeat your previous answer.','Say that again.','Explain your last answer.'])
def test_missing_memory(q):
    assert recall(q,{}) and 'None' not in recall(q,{})


@pytest.mark.parametrize('q',['Why can\'t you read the text?','Is the image blurry?','Help me retake the image.','How should I take a better picture?','Should I move closer?'])
def test_guidance_without_inference(tmp_path,q):
    base,calls=fake_base(tmp_path,[]);im=Image.new('RGB',(600,300));state=prepare(None,im)
    r=turn(base,state,im,q)
    assert not calls and r[3]['decision']=='experimental_capture_guidance'
    assert r[3]['quality']['heuristic'] and 'guarantee' in r[0]


@pytest.mark.parametrize('q,kind,value',[('What is the serving size?','serving size','1 cup (245 g)'),('What is the net weight?','net quantity','20 oz'),('How many calories are there?','calories','110')])
def test_fields(q,kind,value):
    assert object.__new__(FullTextSession).classify(q).task=='targeted'
    assert field_intent(q)==kind
    label={'serving size':'Serving size','net quantity':'Net wt','calories':'Calories'}[kind]
    r=targeted(q,[row(label+': '+value)])
    assert value in r['answer'] and r['supporting_evidence_ids']
    assert targeted(q,[])['answer_source']=='abstention'


def test_quantity_is_not_serving_and_calories_conflict():
    assert targeted('What is the net weight?',[row('Serving size: 1 cup (245 g)')])['answer_source']=='abstention'
    r=targeted('How many calories?',[row('Calories per serving: 110'),row('Calories per container: 250',80)])
    assert r['decision']=='experimental_field_conflicting'
    assert targeted('What is the net weight?',[row('Net wt: 20 oz',confidence=.4)])['answer_source']=='abstention'


@pytest.mark.parametrize('q',['Does the label mention the word sugar?','Does it contain sugar?'])
def test_literal_presence_not_safety(q):
    from reading_relevance import compose
    assert object.__new__(FullTextSession).classify(q).task=='reading'
    assert 'Yes.' in compose(q,{'evidence':[row('Ingredients: oats, sugar')]})['answer']
    assert 'does not establish' in compose(q,{'evidence':[]})['answer']


def test_heating_heading_not_complete_directions():
    from reading_relevance import compose
    assert object.__new__(FullTextSession).classify('How do I heat this?').task=='reading'
    r=compose('How do I heat this?',{'evidence':[row('HEATING INSTRUCTIONS')]})
    assert r['answer_source']=='abstention' and 'heading alone' in r['answer']
    r=compose('How do I heat this?',{'evidence':[row('HEATING INSTRUCTIONS'),row('2%',30),row('Microwave times may vary.',60)]})
    assert r['answer_source']=='abstention'


def test_early_question_survives_recent_display_limit(tmp_path):
    base,calls=fake_base(tmp_path,evidence());im=Image.new('RGB',(800,600));state=prepare(None,im)
    for n in range(9):turn(base,state,im,'Read the first step' if n==0 else 'Read the second step')
    before=list(calls)
    r=turn(base,state,im,'What was my first question?')
    assert r[0]=='Your first question was: Read the first step' and calls==before
    assert len(state['conversation']['turns'])==6
    assert len(state['conversation']['question_history'])==9


def test_scene_candidates_are_explicit_and_do_not_override_abstention():
    from experimental_scene import describe
    rows=[dict(kind='object',id='o1',text='chair',confidence=.9,box=[0,0,50,50])]
    frozen={'final_answer':'room'}
    r=describe('Could you describe this image?',frozen,rows)
    assert 'chair' in r['answer'] and 'candidates' in r['answer'] and frozen=={'final_answer':'room'}
    assert describe('Could you describe this image?',{'final_answer':'unanswerable'},rows) is None
    assert describe('What colour is this?',frozen,rows) is None
    assert describe('What objects are visible?',frozen,[]) is None


def test_short_wrapped_passage_not_cut_after_three_lines():
    from reading_relevance import compose
    rows=[row(s,n*30) for n,s in enumerate(['A visitor describes a journey.', 'They travelled by bus.', 'The journey continued', 'through the town.', 'They reached a station.', 'The account ends there.'])]
    r=compose('Explain the passage.',{'evidence':rows})
    assert 'The account ends there.' in r['answer'] and len(r['supporting_evidence_ids'])==6
    assert 'does not verify' in r['answer']
