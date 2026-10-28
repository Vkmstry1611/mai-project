"""Revalidate final composition/UI via HTTP using preserved real OCR captures.

Only OCR extraction is replayed; speech, Gradio and engine selection wiring are real.
This is explicitly not a new recognition benchmark or GPU inference run.
"""
import json
from pathlib import Path
import socket
import time
import wave

def main():
    from PIL import Image
    from gradio_client import Client,handle_file
    from visionbridge.full_text_session import FullTextSession
    from launch_workspace import build_app,launch_options
    from visionbridge.workspace_ui import ASSISTANT
    from visionbridge.investigate_reading_upgrade import OUT,PHOTOS
    raw=json.loads((OUT/'rapid-baseline.json').read_text(encoding='utf-8'))
    session=FullTextSession('reports/phase4_policy/run-local-20261013-v1')
    original=session.reader.read
    def replay(image,manual=None):
        for value in raw.values():
            if tuple(value['result']['orientation']['original_size'])==image.size:
                return {'result':dict(value['result'],cache_hit=True),'memory':{'capture_replay':True}}
        return original(image,manual)
    session.reader.read=replay
    demo,_=build_app(session.run_dir,session);checks={};records=[]
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        _,url,_=demo.queue().launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options());client=Client(url,verbose=False)
        for q in ('What is written in this image?','Read every visible word.','What website is shown?','What happened in the story?','Summarize this passage.'):
            image=raw['ufo']['path'];start=time.perf_counter()
            r=client.predict(handle_file(image),q,False,ASSISTANT,'0',api_name='/analyze')
            d=next(x for x in r if isinstance(x,dict) and 'final_answer' in x)
            with wave.open(r[1]) as f:assert f.getnframes()>0
            if 'website' in q:
                assert 'nolongerquivering.com' in r[0] and 'cannot confirm' in r[0] and 'Safari' not in r[0]
                assert any(c['visible_url'] and 'http' in c['visible_url'] for c in d['website_evidence'])
            if 'every' in q:assert 'Surrounding interface text:' in r[0] and 'Visible text region:' in r[0]
            if q.startswith('What is written'):assert 'Safari' not in r[0] and 'http' not in r[0]
            if q.startswith('What happened'):assert 'FASTERTHAN' in r[0]
            records.append({'question':q,'text':r[0],'seconds':time.perf_counter()-start,'details':d});checks[q]=True
        client.predict('EasyOCR',api_name='/set_ocr_engine')
        hello='reports/full_text_reading/api-17b203b2/hello.png'
        r=client.predict(handle_file(hello),'Read text',False,ASSISTANT,'0',api_name='/analyze')
        assert r[0]=='HELLO';checks['final-easy-worker-real-hello']=True
        props={c['props'].get('elem_id'):c['props'] for c in demo.config['components']}
        assert props['vb-evidence']['open'] is False
        assert props['vb-playback']['autoplay'] is True and props['vb-mute']['value'] is False
        checks['collapsed-evidence-default-speech']=True
        with (OUT/'FINAL_API_REPLAY.json').open('x',encoding='utf-8') as f:json.dump({'checks':checks,'records':records,'scope':'HTTP cached-real-OCR composition plus real EasyOCR HELLO and real speech; no GPU inference'},f,indent=2,ensure_ascii=False)
        print('Final API checks',len(checks),flush=True)
    finally:demo.close();session.close()

if __name__=='__main__':main()
