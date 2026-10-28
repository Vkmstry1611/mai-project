from dataclasses import replace
import json
from pathlib import Path

import pytest
from PIL import Image

from visionbridge.persistent_demo_worker import FileGuard
from visionbridge.persistent_demo_client import PersistentTransport,ROOT


def test_guard_refuses_changed_and_missing_artifacts(tmp_path):
    file=tmp_path/'model';file.write_text('original');guard=FileGuard([file]);guard.check()
    file.write_text('changed and longer')
    with pytest.raises(RuntimeError,match='changed'):guard.check()
    file.unlink()
    with pytest.raises(RuntimeError,match='disappeared'):guard.check()


def test_transport_preserves_worker_response(tmp_path):
    result={'display_answer':'HELLO','status':'Complete','details':{'baseline_answer':'hello','final_answer':'HELLO'}}
    calls=[]
    class Client:
        startup={'ready':True}
        def infer(self,*args):calls.append(args);return dict(result=result,timings={'inference_seconds':.1})
    timings={};path=tmp_path/'input.png'
    response=PersistentTransport(Client(),False,timings).run(['python',str(ROOT/'phase4_infer.py'),'--image',str(path),'--question','Read this'])
    assert json.loads(response.stdout)==result
    assert calls==[(path,'Read this',False)]
    assert timings['worker_stages']['inference_seconds']==.1
    with pytest.raises(ValueError):PersistentTransport(Client(),False,{}).run(['python','other.py'])


@pytest.mark.parametrize('reading',[False,True])
def test_session_preserves_frozen_response_and_rotation_routing(monkeypatch,tmp_path,reading):
    import app_phase4
    from visionbridge.launch_persistent_demo import Session
    received=[]
    class Client:
        def close(self):received.append('closed')
    details={'final_answer':'HELLO','baseline_answer':'hello'}
    original=('HELLO',None,'Complete',details,None)
    def ask(image,question,speak,run):
        received.append((image.size,app_phase4.subprocess.reading,question));return original
    monkeypatch.setattr(app_phase4,'ask',ask)
    session=Session(tmp_path,Client())
    out=session.request(Image.new('RGB',(120,60)),'Read this',False,reading,'90')
    assert out[:5]==original and out[3] is details
    assert received==[((60,120) if reading else (120,60),reading,'Read this')]
    session.close();assert received[-1]=='closed'


def test_ocr_cache_reuse_image_identity_and_corruption(tmp_path):
    from config import Config
    from modules import evidence_store
    calls=[]
    class OCR:
        @staticmethod
        def extract(image,config):
            calls.append(image.size)
            return [dict(text='HELLO',confidence=.99,box=[0,0,20,10])]
    cfg=replace(Config(),device='cpu',ocr_device='cpu',cache_dir=tmp_path/'weights',evidence_cache_dir=tmp_path/'evidence')
    first,status=evidence_store.extract(Image.new('RGB',(100,50),'white'),'ocr',OCR,cfg)
    second,again=evidence_store.extract(Image.new('RGB',(100,50),'white'),'ocr',OCR,cfg)
    assert first==second and again['cache_hit'] and len(calls)==1
    _,different=evidence_store.extract(Image.new('RGB',(100,50),'black'),'ocr',OCR,cfg)
    assert different['cache_key']!=status['cache_key'] and len(calls)==2
    path=cfg.evidence_cache_dir/(status['cache_key']+'.json')
    payload=json.loads(path.read_text());payload['record']['items'][0]['text']='WRONG'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='integrity'):evidence_store.extract(Image.new('RGB',(100,50),'white'),'ocr',OCR,cfg)
    assert len(calls)==2  # Never silently replace a corrupt cache.


def test_repeat_statistics_are_exact():
    from scripts.benchmarks.benchmark_persistent_demo import summarize
    assert summarize([2,4,3])==dict(individual_seconds=[2,4,3],mean_seconds=3,median_seconds=3)


def test_shutdown_targets_only_owned_worker():
    from visionbridge.persistent_demo_client import WorkerClient
    class Pipe:
        def __init__(self):self.messages=[];self.closed=False
        def write(self,text):self.messages.append(text)
        def flush(self):pass
        def close(self):self.closed=True
    class Process:
        def __init__(self):self.stdin=Pipe();self.stdout=Pipe();self.waited=False
        def poll(self):return None if not self.waited else 0
        def wait(self,timeout):self.waited=True
    client=WorkerClient('.')
    client.process=Process();client.close()
    assert client.process.stdin.messages==['{"action":"shutdown"}\n']
    assert client.process.waited and client.process.stdin.closed and client.process.stdout.closed


def test_each_route_preserves_its_original_cpu_thread_configuration(tmp_path):
    from types import SimpleNamespace
    from visionbridge.persistent_demo_worker import Engine
    seen=[]
    cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None,manual_seed_all=lambda seed:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,get_device_name=lambda index:'test GPU')
    engine=Engine.__new__(Engine);engine.root=tmp_path;engine.reading_threads=12
    engine.torch=SimpleNamespace(cuda=cuda,set_num_threads=seen.append,manual_seed=lambda seed:None)
    engine.guard=SimpleNamespace(check=lambda:None);engine.ocr=object()
    evidence=[dict(id='ocr-hello',kind='ocr',text='HELLO',box=[0,0,10,10],confidence=.99,status='complete')]
    engine.evidence_store=SimpleNamespace(extract=lambda *a:(evidence,{'status':'complete','cache_hit':False}))
    engine.pipeline=SimpleNamespace(run=lambda *a:('hello',None,'Complete',{'warnings':[]},None))
    photo=tmp_path/'synthetic.png';Image.new('RGB',(20,20)).save(photo)
    reading,_=engine.infer(photo,'Read this',True)
    frozen,_=engine.infer(photo,'Read this',False)
    assert seen==[12,4]
    assert reading['display_answer']=='Recognized text fragments: HELLO.'
    assert frozen['display_answer']=='hello'
