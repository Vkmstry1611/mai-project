import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from visionbridge.website_evidence import candidates,decide
from visionbridge.reading_relevance import compose
from visionbridge.full_text_session import FullTextSession
from visionbridge.image_conversation import prepare,resolve,task_quality,MAX_TURNS

def row(text,y=0,confidence=.99):
    return {'id':text+str(y),'text':text,'confidence':confidence,'box':[0,y,500,y+25],'corrected_box':[0,y,500,y+25],
        'corrected_polygon':[[0,y],[500,y],[500,y+25],[0,y+25]],'source':{'engine':'RapidOCR'},'kind':'ocr'}

@pytest.mark.parametrize('q',['What website is shown?','What is the website in this image?','What is website in the image?',
    'Which site is open?','What is the URL?','Can you read the website address?','Which domain is visible?'])
def test_website_variations(q):
    base=object.__new__(FullTextSession);assert base.classify(q).task=='reading'
    r=compose(q,{'evidence':[row('http://example.com/image.gif')]})
    assert 'example.com' in r['answer'] and 'cannot confirm' in r['answer']
    assert r['supporting_evidence_ids']

def test_url_parsing_qualifies_source_and_never_repairs_missing_punctuation():
    d=decide('What website is shown?',[row('https://example.com/story.gif')])
    assert d['candidates'][0]['role']=='embedded_image_source_url'
    assert candidates([row('example . com')])[0]['domain']=='example.com'
    assert candidates([row('http:/example.com')])[0]['visible_url'] is None
    assert not candidates([row('https://example com')])
    assert not candidates([row('skeptic.gif')])
    assert candidates([row('http://exam0le.com')])[0]['domain']=='exam0le.com'
    assert candidates([row('name@example.com')])[0]['role']=='email_domain'
    assert 'email address' in decide('What website?', [row('name@example.com')])['answer']
    assert candidates([row('https://exam ple.com')])[0]['role']=='partial_domain_fragment'
    assert decide('What website?',[])['reason']=='no_reliable_website_evidence'
    assert decide('What website?',[row('Page title: Demo')])['reason']=='labeled_title_without_domain'
    assert candidates([row('Visit example.com.')])[0]['domain']=='example.com'
    assert candidates([row('example.com.evil.net')])[0]['domain']=='example.com.evil.net'

def test_broken_lines_require_literal_dot_and_near_geometry():
    rows=[row('https://example.',0),row('com/image.gif',30)]
    d=decide('What is the URL?',rows);assert 'https://example.com/image.gif' in d['answer']
    assert len(d['ids'])==2
    assert not decide('What website?', [row('https://example.',0),row('com/image.gif',500)])['ids']

def test_conflicting_urls_and_confidence_are_preserved():
    d=decide('Which website?',[row('https://a.com'),row('https://b.org',40),row('https://bad.net',80,.4)])
    assert d['reason']=='multiple_visible_domains' and 'bad.net' not in d['answer']
    assert len(d['candidates'])==3 and len(d['ids'])==2

def fake_base(tmp_path,rows):
    base=object.__new__(FullTextSession);base.run_dir=tmp_path;base.engine_choice='Automatic';calls=[]
    def read(image,angle):
        calls.append('ocr')
        return {'result':{'evidence':copy.deepcopy(rows),'source':{'engine':'RapidOCR'},'transcription':'raw complete',
            'orientation':{'selected_ccw_degrees':0},'cache_hit':False,'request_seconds':1.,'ocr_lines':len(rows)},'memory':{}}
    base.reader=SimpleNamespace(read=read,startup={});base.easy=SimpleNamespace(read=read,startup={})
    def vqa(*args):
        calls.append('blip')
        return ('Original abstention',None,'Complete',{'baseline_answer':'unanswerable','final_answer':'unanswerable','answer_source':'abstention',
            'evidence':[],'selected_evidence':[],'supporting_evidence_ids':[],'phase3_gate_reason':'Phase 2 already abstained'},None,'',
            {'callback_seconds':1.,'worker_stages':{'blip_seconds_including_loading':.8}})
    base.legacy=SimpleNamespace(request=vqa)
    return base,calls

def turn(base,state,image,q):
    facade=base.for_conversation(state,image);route=facade.classify(q)
    return facade.request_auto(image,q,False,True) if route.task in ('reading','targeted') else facade.request(image,q,False,False,'0')

def test_multiple_questions_use_complete_raw_cache_not_prior_answer(tmp_path):
    rows=[row('http://example.com/image.gif'),row('A visitor saw a bright light.',150),row('The visitor went home.',185)]
    base,calls=fake_base(tmp_path,rows);image=Image.new('RGB',(600,300));state=prepare(None,image)
    first=turn(base,state,image,'Read the passage.');assert 'http' not in first[0]
    second=turn(base,state,image,'what is website in the image?');assert 'example.com' in second[0]
    assert not second[3]['quality']['capture_suggestion_needed'] and not second[3]['quality']['warnings']
    assert not second[3]['local_quality']['warnings']
    third=turn(base,state,image,'Detect every text');assert 'http' in third[0]
    assert calls==['ocr'] and second[6]['conversation_evidence_cache_hit']
    assert len(third[3]['evidence'])==3

def test_visual_cache_preserves_frozen_abstention_and_no_ocr(tmp_path):
    base,calls=fake_base(tmp_path,[]);image=Image.new('RGB',(600,300));state=prepare(None,image)
    a=turn(base,state,image,'What is in the image?');b=turn(base,state,image,'What is in the image?')
    assert calls==['blip']
    assert a[3]['final_answer']==b[3]['final_answer']=='unanswerable'
    assert a[3]['frozen_display_answer']=='Original abstention' and a[3]['assistant_trace']['blip_inference_observed']
    assert b[3]['assistant_trace']['visual_cache_hit'] and not b[3]['assistant_trace']['blip_inference_observed']
    assert 'person holding' not in a[0]

def test_visual_cache_invalidates_if_artifact_changes(tmp_path,monkeypatch):
    from visionbridge import image_conversation
    monkeypatch.setattr(image_conversation,'ARTIFACT_ROOT',tmp_path)
    folder=tmp_path/'models/blip_vizwiz_v2';folder.mkdir(parents=True)
    fixture=folder/'model.safetensors';fixture.write_bytes(b'synthetic fixture, not a model')
    base,calls=fake_base(tmp_path,[]);image=Image.new('RGB',(600,300));state=prepare(None,image)
    turn(base,state,image,'What is in this image?');turn(base,state,image,'What is in this image?')
    assert calls==['blip']
    fixture.write_bytes(b'changed synthetic fixture')
    turn(base,state,image,'What is in this image?');assert calls==['blip','blip']

def test_context_switch_expiry_and_isolation():
    a=Image.new('RGB',(20,20),'red');b=Image.new('RGB',(20,20),'blue');state=prepare(None,a)
    state['conversation']['turns']=[{'question':'old','support_ids':['x']}];state['conversation']['ocr']={'private':'data'}
    assert not prepare(state,b)['conversation']['ocr']
    assert prepare(state,a)['conversation']['ocr']
    state['conversation']['updated']=0;assert not prepare(state,a)['conversation']['turns']
    assert not prepare(None,a)['conversation']['turns']

def test_references_retake_and_clear_do_not_invent_facts():
    assert resolve('Can you read it?',{})[0].startswith('Read the visible')
    assert resolve('Explain that',{})[1]
    assert resolve('Read the rest',{})[1]
    a=dict(row('This is a first line.'),id='a');b=dict(row('This is a second line.',40),id='b')
    c={'turns':[{'scope':'main','support_ids':['a','b']}],'read_ids':['a','b'],'ocr':{'cache':{'result':{'evidence':[a,b]}}}}
    assert resolve('Explain that',c)[0]=='Summarize this passage.'
    assert resolve('Read the rest',c)[2]==['a','b']
    assert resolve('Explain that',{'turns':[{'scope':'main','support_ids':['invented']}],'ocr':{}})[1]
    assert not prepare(None,Image.new('RGB',(20,20)))['conversation']['turns']

def test_read_rest_stays_in_original_scope(tmp_path):
    base,calls=fake_base(tmp_path,[row('http://example.com'),row('The passage first line.',100),row('The passage second line.',140)])
    image=Image.new('RGB',(600,300));state=prepare(None,image)
    turn(base,state,image,'Read the passage');rest=turn(base,state,image,'Read the rest')
    assert 'no additional' in rest[0] and 'http' not in rest[0] and calls==['ocr']

def test_task_quality_never_blocks_and_history_is_bounded(tmp_path):
    image=Image.new('RGB',(100,100),'black')
    for task in ('reading','vqa'):assert not task_quality(image,task)['blocks_inference']
    base,calls=fake_base(tmp_path,[]);state=prepare(None,image)
    for n in range(10):turn(base,state,image,'What is in this image?')
    assert len(state['conversation']['turns'])==MAX_TURNS and calls==['blip']
    result=turn(base,state,image,'What should I do to make it clearer?')
    assert result[3]['decision']=='experimental_capture_guidance' and calls==['blip']

def test_exact_product_needs_printed_label(tmp_path):
    base,calls=fake_base(tmp_path,[row('random words')]);image=Image.new('RGB',(600,300));state=prepare(None,image)
    result=turn(base,state,image,'What exact product is this?')
    assert result[3]['answer_source']=='abstention' and calls==['ocr']

def test_verified_bundled_weights_are_reused_without_download(tmp_path,monkeypatch):
    import hashlib
    from visionbridge import assistant_runtime
    folder=tmp_path/'reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/model-cache/easyocr';folder.mkdir(parents=True)
    p=folder/'fixture.pth';p.write_bytes(b'fixture')
    monkeypatch.setattr(assistant_runtime,'WEIGHTS',{'fixture.pth':hashlib.sha256(b'fixture').hexdigest()})
    assert assistant_runtime.easyocr_directory(tmp_path)==folder
    p.write_bytes(b'changed')
    with pytest.raises(FileNotFoundError,match='no automatic model download'):assistant_runtime.easyocr_directory(tmp_path)

def test_image_change_handler_immediately_invalidates_context_without_question_output(tmp_path):
    from launch_workspace import build_app
    base,calls=fake_base(tmp_path,[]);base.close=lambda:None
    demo,_=build_app(tmp_path,base)
    image=Image.new('RGB',(100,100),'red');old=prepare(None,image);old['conversation']['ocr']={'private':'old'}
    fn=next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__=='upload_reset')
    result=fn(Image.new('RGB',(100,100),'blue'),old)
    assert not result[12]['conversation']['ocr'] and not result[12]['conversation']['turns']
    assert len(result)==14
