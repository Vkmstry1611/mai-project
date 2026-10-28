"""Pure development fixtures; never read holdout data."""
import copy
import pytest
from types import SimpleNamespace
from PIL import Image
from visionbridge.reading_relevance import compose,intent,group_regions
from visionbridge.full_text_session import FullTextSession

def row(text,x,y,w=400,h=25,score=.99,engine='RapidOCR'):
    return dict(id=text+str(y),text=text,confidence=score,box=[x,y,x+w,y+h],corrected_box=[x,y,x+w,y+h],source={'engine':engine})

def fixture():
    return {'evidence':[row('Safari File Edit View History',0,0,500),row('https://example.com/story.png',0,35,500),
        row('A visitor saw a bright light.',0,150,650,40),row('The visitor went back home.',0,200,650,40)],'transcription':'original immutable transcription'}

@pytest.mark.parametrize('q,task',[('Read every visible word.','all'),('Read all the words.','all'),('What website is shown?','interface'),
    ('Summarize this passage.','summary'),('What happened in the story?','text_question'),('What is written?','main')])
def test_intents(q,task):assert intent(q)==task

def test_main_excludes_browser_preserves_raw_and_speech_text():
    data=fixture();before=copy.deepcopy(data);r=compose('What is written?',data)
    assert 'visitor' in r['answer'] and 'Safari' not in r['answer'] and 'https' not in r['answer']
    assert data==before and r['evidence']==data['evidence'] and len(r['excluded_evidence'])==2
    assert r['answer']==r['final_answer'] and '[unreadable' not in r['answer']

def test_read_all_and_interface():
    assert 'Safari' in compose('Read every visible word',fixture())['answer']
    r=compose('What website is shown?',fixture())
    assert 'example.com' in r['answer'] and 'cannot confirm' in r['answer'] and 'visitor' not in r['answer']
    assert r['website_evidence'][0]['visible_url']=='https://example.com/story.png'
    assert r['website_evidence'][0]['role']=='embedded_image_source_url'

def test_summary_extracts_without_invention_specific_question_abstains():
    r=compose('Summarize this passage',fixture());assert 'The passage describes' in r['answer'] and 'visitor' in r['answer']
    assert compose('Why did he leave the story?',fixture())['answer_source']=='abstention'

def test_symbols_digits_hello_and_confidence_do_not_guess():
    data={'evidence':[row('HELLO',0,0),row('12.5%',0,50),row('*',0,100,score=.3)],'transcription':'raw'}
    r=compose('Read text',data);assert 'HELLO' in r['answer'] and '12.5%' in r['answer'] and '*' not in r['answer']
    assert sum(r['answer'].count(s) for s in ['Some smaller'])==1

def test_remote_columns_and_ambiguity():
    a=[row('This is the left passage.',0,y) for y in (0,40,80)]
    b=[row('This is the right passage.',1000,y) for y in (0,40,80)]
    assert len(group_regions(a+b))==2
    assert 'multiple possible' in compose('Read passage',{'evidence':a+b})['answer']

def test_easy_original_threshold_and_route():
    r=compose('Read text',{'evidence':[row('unsafe',0,0,score=.89,engine='EasyOCR')]})
    assert r['answer_source']=='abstention'
    s=object.__new__(FullTextSession)
    for q in ('Summarize this passage','What website is shown?','What happened in the story?'):assert s.classify(q).task=='reading'
    assert s.classify('What is in this image?').task=='vqa'
    assert s.classify('What color is the website background?').task=='vqa'
    assert s.classify('What is in this website image?').task=='vqa'

def test_orchestration_is_selective_and_keeps_primary_evidence():
    calls=[];s=object.__new__(FullTextSession);s.engine_choice='Automatic'
    def reader(name,confidence):
        def read(image,angle):
            calls.append(name)
            r=row('HELLO',0,0,w=80,h=20,score=confidence,engine=name);r['corrected_polygon']=[[0,0],[80,0],[80,20],[0,20]]
            return {'result':{'evidence':[r],'source':{'engine':name},'transcription':'HELLO','orientation':{'selected_ccw_degrees':0},'cache_hit':False},'memory':{}}
        return SimpleNamespace(read=read,startup={})
    s.reader=reader('RapidOCR',.99);s.easy=reader('EasyOCR',.99)
    first=s.full_request(Image.new('RGB',(100,50)),'Read text')
    assert calls==['RapidOCR'] and first[3]['final_answer']=='HELLO'
    s.reader=reader('RapidOCR',.2)
    result=s.full_request(Image.new('RGB',(100,50)),'Read text')
    assert calls[-2:]==['RapidOCR','EasyOCR']
    assert result[3]['alternate_engine_evidence']['evidence'][0]['confidence']==.2
    assert result[3]['engine_selection']['selected']=='EasyOCR'

def test_raw_cache_verified_and_question_independent(tmp_path):
    from visionbridge.full_text_ocr import FullTextEngine
    engine=object.__new__(FullTextEngine);engine.source={'engine':'fixture'}
    calls=[]
    def fake(*args,**kwargs):calls.append(True);return [([[0,0],[60,0],[60,20],[0,20]],'HELLO',.99)],[]
    engine.engine=fake;engine.orientation=lambda image:(0,{})
    image=Image.new('RGB',(100,50),'white')
    first=engine.read(image,tmp_path);second=engine.read(image,tmp_path)
    assert len(calls)==1 and second['cache_hit']
    assert compose('Read text',second)['evidence']==first['evidence']
    path=next(tmp_path.glob('*.json'));path.write_text(path.read_text().replace('HELLO','FORGED'))
    with pytest.raises(ValueError,match='integrity'):engine.read(image,tmp_path)

def test_targeted_metadata_does_not_mark_other_fields_supporting():
    s=object.__new__(FullTextSession);s.engine_choice='RapidOCR'
    rows=[row('EXP: 12/2027',0,0),row('PRICE: Rs 99.00',0,60)]
    response={'result':{'evidence':rows,'source':{'engine':'RapidOCR'},'transcription':'raw','orientation':{'selected_ccw_degrees':0},'cache_hit':False},'memory':{}}
    s.reader=SimpleNamespace(read=lambda *args:response,startup={})
    r=s.full_request(Image.new('RGB',(500,150)),'What is the expiry date?',target_field=True)[3]
    assert len(r['selected_evidence'])==1 and len(r['excluded_evidence'])==1
    assert r['selected_evidence'][0]['text']=='EXP: 12/2027'
