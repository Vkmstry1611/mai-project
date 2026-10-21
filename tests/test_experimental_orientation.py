"""Orientation contracts without neural models or evaluation data."""
import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from experimental_orientation import (ANGLES, THRESHOLD, assess, bounds, mapped_items, matrices,
    rank, read, transform, VERSION)


def item(text='HELLO', confidence=.99, box=(10, 10, 110, 40), identity='hello'):
    x, y, z, t = box
    return dict(id=identity, text=text, confidence=confidence, box=list(box),
        polygon=[[x,y],[z,y],[z,t],[x,t]], kind='ocr', status='complete')


@pytest.mark.parametrize('angle', ANGLES)
def test_transform_and_matrix_round_trip(angle):
    for p in [(0,0), (100,0), (100,60), (0,60), (32.5,18.2)]:
        assert transform(transform(p,100,60,angle),100,60,angle,True)==pytest.approx(p)
    m=matrices((200,120),(100,60),angle)
    def apply(a,p):return [a[0][0]*p[0]+a[0][1]*p[1]+a[0][2],a[1][0]*p[0]+a[1][1]*p[1]+a[1][2]]
    assert apply(m['inverse_3x3'],apply(m['forward_3x3'],[65,37]))==pytest.approx([65,37])


@pytest.mark.parametrize('angle', ANGLES)
def test_all_polygon_corners_preserved_with_scale(angle):
    row=item(box=(10,10,50,30))
    before=copy.deepcopy(row)
    result=mapped_items(dict(angle=angle,items=[row]),(200,120),(100,60),'original')[0]
    assert result['corrected_box']==row['box'] and row==before
    assert result['box']==bounds(result['original_polygon'])
    for p,q in zip(result['original_polygon'],result['corrected_polygon']):
        assert transform((p[0]/2,p[1]/2),100,60,angle)==pytest.approx(q)


def test_many_confident_single_glyphs_cannot_beat_real_words():
    noise=[item(str(i%10),identity=str(i)) for i in range(100)]
    good=[item()]
    assert not assess(noise)['eligible']
    assert assess(good)['eligible']
    assert assess(good)['score']>assess(noise)['score']


def test_numerical_text_is_not_banned_and_no_probability_claim():
    assert assess([item('2026')])['eligible']
    assert THRESHOLD==.90
    assert 'probability' not in assess([item()])


def test_tie_and_weak_text_abstain():
    c=[dict(angle=a,assessment=assess([item()])) for a in (0,90)]
    assert rank(c)==(None,'ambiguous_or_mixed_orientation')
    assert rank([dict(angle=0,assessment=assess([item(confidence=.8)]))])[0] is None


class FakeReader:
    calls=0
    def readtext(self,array):
        self.calls+=1
        # First orientation is reliable. Other rotations remain weak.
        return [([[10,10],[110,10],[110,40],[10,40]],'HELLO',.99 if self.calls==1 else .1)]
    def recognize(self,*a,**kw):return [(None,'HELLO',.99)]


def test_read_cache_version_raw_trace_no_preprocessing_and_integrity(tmp_path):
    image=Image.new('RGB',(200,180),'white');reader=FakeReader()
    result=read(image,tmp_path,{'version':'test'},lambda:reader)
    assert result['orientation']['selected_ccw_degrees']==0
    assert result['answer']=='Recognized text fragments: HELLO.'
    assert len(result['orientation']['all_candidates'])==4 and reader.calls==4
    assert result['orientation']['estimator_used_to_skip_angles'] is False
    assert result['evidence'][0]['corrected_box']==[10.,10.,110.,40.]
    hit=read(image,tmp_path,{'version':'test'},lambda:(_ for _ in ()).throw(AssertionError('cache miss')))
    assert hit['orientation']['cache_hit'] and hit['extraction_status']['ocr']['cache_hit']
    assert hit['answer']==result['answer']
    file=next((tmp_path/VERSION).glob('*.json'));data=json.loads(file.read_text())
    data['result']['answer']='invented';file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='integrity mismatch'):read(image,tmp_path,{'version':'test'},lambda:reader)


def test_time_budget_abstains_without_caching_incomplete_search(tmp_path):
    result=read(Image.new('RGB',(200,180)),tmp_path,{},FakeReader,budget=0)
    assert result['answer_source']=='abstention'
    assert result['orientation']['selected_ccw_degrees'] is None
    assert result['orientation']['timed_out']
    assert not list(tmp_path.rglob('*.json'))


def test_verification_failure_never_uses_blip_or_guesses(tmp_path):
    reader=FakeReader();reader.recognize=lambda *a,**kw:[(None,'OTHER',.99)]
    result=read(Image.new('RGB',(200,180)),tmp_path,{},lambda:reader)
    assert result['answer_source']=='abstention' and not result['supporting_evidence_ids']
    assert result['orientation']['decision_reason']=='orientation_verification_failed'


def test_optional_workspace_hook_only_used_for_automatic_reading():
    from workspace_ui import WorkspaceController,ASSISTANT,VISUAL,READING
    calls=[]
    class Session:
        def request(self,*a):calls.append(('old',a));return ('HELLO',None,'Complete',{},None,'',{})
        def request_auto(self,*a):calls.append(('auto',a));return ('HELLO',None,'Complete',{},None,'',{})
    controller=WorkspaceController(Session())
    for mode,q in [(ASSISTANT,'Read this'),(READING,'Read this'),(VISUAL,'What is this?'),(ASSISTANT,'What color is this?')]:
        list(controller.analyze(Image.new('RGB',(10,10)),q,False,mode,'90'))
    assert [c[0] for c in calls]==['auto','old','old','old']
    assert calls[1][1][-1]=='90'


def test_session_manual_and_disabled_routes_return_identical_values(monkeypatch):
    from launch_persistent_demo import Session
    from orientation_workspace_session import OrientationSession
    from types import SimpleNamespace
    original=('original',None,'Complete',{},None,'',{})
    seen=[]
    def request(self,*a,**kw):seen.append((self.client.automatic,a));return original
    monkeypatch.setattr(Session,'request',request)
    s=OrientationSession('.',client=SimpleNamespace(automatic=False))
    assert s.request('image','Read this',False,True,'90') is original
    assert s.request_auto('image','What color?',False,False) is original
    assert [a[0] for a in seen]==[False,False]
