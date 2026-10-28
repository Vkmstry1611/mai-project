import copy
import pytest
from visionbridge.reading_relevance import compose
from visionbridge.text_hierarchy import request
from test_reading_relevance import row


def evidence():
    return [row('Brown ground beef',0,0),row('in a skillet; drain.',0,30),
            row('Stir in water.',0,90),row('Heat to boiling.',0,120),
            row('Reduce heat.',0,180),row('Cover and simmer.',0,210)]


@pytest.mark.parametrize('q,yes,no',[
    ('Read the first step','Brown','Stir'),('Read the second cooking step','Stir','Brown'),
    ('Repeat step three','Reduce','Stir'),('Read the first paragraph','Brown','Stir'),
    ('Read the second paragraph','Stir','Brown'),('Continue from the second step','Reduce','Brown')])
def test_specific(q,yes,no):
    d={'evidence':evidence()};before=copy.deepcopy(d);r=compose(q,d)
    assert yes in r['answer'] and no not in r['answer'];assert d==before
    assert r['supporting_evidence_ids'];assert 'not recognized' in r['answer'] or 'step' not in q


def test_label_is_not_step_and_second_label_clarifies():
    assert 'cannot reliably distinguish' in compose('I want first label to be read only',{'evidence':evidence()})['answer']
    r=compose('Read the second label',{'evidence':evidence()})
    assert not r['supporting_evidence_ids'] and 'cannot reliably distinguish' in r['answer']


def test_no_spacing_no_invented_steps():
    rows=[row('Stir in water.',0,0),row('Heat to boiling.',0,30),row('Cover and simmer.',0,60)]
    r=compose('Read the second step',{'evidence':rows})
    assert not r['supporting_evidence_ids'] and 'cannot reliably select' in r['answer']


def test_read_all_unchanged_and_stable_ids():
    a=compose('Read all text',{'evidence':evidence()});b=compose('Read the second step',{'evidence':evidence()})
    assert 'Brown' in a['answer'] and 'Reduce' in a['answer']
    assert a['text_hierarchy']==b['text_hierarchy']


def test_heading_and_rest():
    rows=[row('Cooking instructions',0,-80)]+evidence()
    assert 'Cooking instructions' in compose('Read the heading',{'evidence':rows})['answer']
    first=compose('Read the first step',{'evidence':rows})
    rest=compose('Read the remaining instructions',{'evidence':rows},omit_ids=first['supporting_evidence_ids'])
    assert 'Brown' not in rest['answer'] and 'Reduce' in rest['answer']


def test_router_spec():
    assert request('Read the second cooking step')['number']==2
    assert request('I want first label to be read only')['kind']=='label'
    assert request('Read the labels') is None


def test_missing_printed_number_does_not_renumber():
    rows=[row('1. Stir water.',0,0),row('Heat slowly.',0,30),
          row('3. Reduce heat.',0,90),row('Cover and simmer.',0,120)]
    assert 'cannot uniquely verify' in compose('Read step two',{'evidence':rows})['answer']
    assert 'Reduce' in compose('Repeat step three',{'evidence':rows})['answer']


def test_history_has_roles_spacing_full_text_and_escaping():
    from visionbridge.image_conversation import history_html
    answer='a'*700+'\n<b>tail</b>'
    h=history_html({'conversation':{'turns':[{'question':'<q>','answer':answer}]}})
    assert 'User:' in h and 'VisionBridge:' in h and 'tabindex="0"' in h
    assert '&lt;b&gt;tail' in h and 'a'*700 in h and '<q>' not in h


def test_followup_reuses_cache_and_scopes_paragraph(tmp_path):
    from test_image_conversation import fake_base,turn
    from visionbridge.image_conversation import prepare
    from PIL import Image
    base,calls=fake_base(tmp_path,evidence());im=Image.new('RGB',(800,600));state=prepare(None,im)
    turn(base,state,im,'Read the cooking instructions')
    second=turn(base,state,im,'Read the second paragraph')
    assert 'Stir' in second[0] and 'Brown' not in second[0]
    first=turn(base,state,im,'Read the first step')
    rest=turn(base,state,im,'Read the rest')
    assert 'Brown' not in rest[0] and 'Reduce' in rest[0]
    assert calls==['ocr']
