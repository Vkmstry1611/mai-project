import copy
import pytest

from experimental_reading import is_reading_request, reconstruct, route_request


def item(text, box, confidence=.99, identity=None):
    return dict(id=identity or text, text=text, box=box, confidence=confidence, kind='ocr', status='complete')


@pytest.mark.parametrize('question', ['What is written in this image?', 'What words are written in this image?',
    'Could you please read this to me?', 'Can you read this?', 'Read the label'])
def test_explicit_reading(question):
    assert is_reading_request(question)


@pytest.mark.parametrize('question', ['What is this?', 'What brand is this?', 'Translate the text',
    'What does this text mean?', 'What colour is the writing?'])
def test_not_literal_reading(question):
    assert not is_reading_request(question)


def test_spatial_order_and_separate_regions_are_not_competing_answers():
    rows=[item('California',[0,40,100,60]), item('Cabernet',[0,0,80,20]),
          item('Sauvignon',[85,0,180,20]), item('NOTICE',[350,0,430,20])]
    before=copy.deepcopy(rows)
    out=reconstruct(rows)
    assert out['lines']==['Cabernet Sauvignon', 'NOTICE', 'California']
    assert out['answer'].startswith('Recognized text fragments:')
    assert rows==before


def test_weak_fragments_do_not_veto_strong_or_get_invented():
    out=reconstruct([item('Sauteed',[0,0,80,20]),item('Potato',[85,0,140,20],.821),
                     item('Bacon',[150,0,220,20])])
    assert out['lines']==['Sauteed', 'Bacon']  # Do not bridge a large missing word.
    assert 'Potato' not in out['answer']
    assert out['partial'] and out['supporting_evidence_ids']==['Sauteed','Bacon']


def test_no_weak_negation_join():
    out=reconstruct([item('DO',[0,0,30,20]),item('NOT',[32,0,62,20],.4),item('ENTER',[64,0,130,20])])
    assert out['lines']==['DO [unclear] ENTER']


def test_overlap_conflict_is_omitted_without_losing_other_region():
    out=reconstruct([item('STOP',[0,0,100,20]),item('SHOP',[0,0,100,20]),item('NOTICE',[0,80,100,100])])
    assert out['lines']==['NOTICE']
    assert len(out['conflicting_ids'])==2


@pytest.mark.parametrize('rows', [[],[item('noise',[0,0,30,20],.2)], [item('bad',[0,0,0,20])]])
def test_abstain_without_blip_fallback(rows):
    assert reconstruct(rows)['decision']=='experimental_reading_abstained'


@pytest.mark.parametrize('enabled,question', [(False,'Read this'),(True,'What is this?')])
def test_frozen_route_returns_original_object(enabled,question):
    original=('salt lake city',None,'Complete',{'frozen':True},None)
    called=[]
    def frozen(): called.append('frozen'); return original
    def reading(): raise AssertionError('Experimental extraction must not run')
    assert route_request(enabled,question,frozen,reading) is original
    assert called==['frozen']


def test_explicit_opt_in_can_read_despite_blip_abstention():
    assert route_request(True,'Read this',lambda:None,lambda:'Cabernet')=='Cabernet'


def test_controller_off_preserves_frozen_values_and_ignores_rotation(monkeypatch,tmp_path):
    from PIL import Image
    import app_phase4
    from launch_reading_demo import request
    original=('unanswerable',None,'Complete',{'baseline_answer':'unanswerable'},None)
    photo=Image.new('RGB',(120,60),'white');seen=[]
    def ask(image,question,speak,run_dir):
        seen.append((image.size,question,speak,run_dir));return original
    monkeypatch.setattr(app_phase4,'ask',ask)
    response=request(photo,'Read this',False,False,'90',tmp_path)
    assert response[:5]==original and response[3] is original[3]
    assert seen==[((120,60),'Read this',False,tmp_path)]
    assert response[-1]['experimental_reading_active'] is False


def test_explicit_rotation_uses_rotated_image_coordinates(monkeypatch,tmp_path):
    from PIL import Image
    import app_phase4
    from launch_reading_demo import request
    seen=[]
    def ask(image,*args):
        seen.append(image.size);return ('text',None,'Experimental',{},None)
    monkeypatch.setattr(app_phase4,'ask',ask)
    out=request(Image.new('RGB',(120,60)),'Read this',False,True,'90',tmp_path)
    assert seen==[(60,120)] and out[-1]['rotation_ccw_degrees']==90


def test_duplicate_box_does_not_repeat_words():
    out=reconstruct([item('Cabernet',[0,0,100,20],identity='one'),item('Cabernet',[0,0,100,20],identity='two')])
    assert out['lines']==['Cabernet']


def test_different_font_sizes_do_not_reorder_vertical_regions():
    out=reconstruct([item('HEADER',[0,0,200,40]),item('caption',[0,100,40,110])])
    assert out['lines']==['HEADER','caption']


def test_hello_output_is_preserved_exactly():
    assert reconstruct([item('HELLO',[10,10,110,35])])['answer']=='Recognized text fragments: HELLO.'


def test_right_word_detected_slightly_higher_still_joins_left_to_right():
    rows=[item('Bacon',[150,8,220,28]),item('Sauteed',[0,10,80,30]),item('Potato',[85,9,145,29])]
    assert reconstruct(rows)['lines']==['Sauteed Potato Bacon']
    assert reconstruct(list(reversed(rows)))==reconstruct(rows)


def test_separate_rows_and_columns_are_not_joined():
    rows=[item('TOP',[0,0,60,20]),item('RIGHT',[400,0,480,20]),item('BOTTOM',[0,40,90,60])]
    assert reconstruct(rows)['lines']==['TOP','RIGHT','BOTTOM']


def test_sloping_chain_does_not_bridge_to_a_different_line():
    rows=[item('ONE',[0,0,40,20]),item('TWO',[45,8,85,28]),item('THREE',[90,16,150,36])]
    assert reconstruct(rows)['lines']==['ONE TWO','THREE']


@pytest.mark.parametrize('enabled',[False,True])
def test_worker_uses_utf8_and_only_opt_in_selects_reading(monkeypatch,tmp_path,enabled):
    from PIL import Image
    import app_phase4
    import launch_reading_demo as demo
    calls=[]
    def process(command,**options):
        calls.append((command,options))
        assert options['env']['PYTHONIOENCODING']=='utf-8'
        return object()
    monkeypatch.setattr(demo.subprocess,'run',process)
    def ask(*a):
        app_phase4.subprocess.run(['python',str(demo.ROOT/'phase4_infer.py'),'--image',str(tmp_path/'input.png')])
        return ('Sautéed',None,'Complete',{},None)
    monkeypatch.setattr(app_phase4,'ask',ask)
    out=demo.request(Image.new('RGB',(20,10)),'Read this',False,enabled,'0',tmp_path)
    assert out[0]=='Sautéed'
    assert ('--reading' in calls[0][0]) is enabled
    assert calls[0][0][1]==str(demo.ROOT/'reading_demo_worker.py')
