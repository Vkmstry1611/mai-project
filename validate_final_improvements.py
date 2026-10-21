"""Small manual-fixture HTTP validation; no dataset evaluation or training."""
import argparse
import json
from pathlib import Path
import socket
import time


def main(out,replay_from=None):
    from gradio_client import Client,handle_file
    from launch_workspace import build_app,launch_options
    from full_text_session import FullTextSession
    from workspace_ui import ASSISTANT
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    sources=json.loads(Path('reports/final_improvements/run-20261021-v1/candidate-paths.json').read_text())
    fixtures={
      'cooking':('reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/data/images/VizWiz_val_00002603.jpg',
          ['Read full text','Read the first step','Read the second step','Read the third step','Could you repeat the previous step?',
           'Go back one step','Read the next step','What ingredients are mentioned?','What is this image about?','Could you repeat my second question?']),
      'ufo':('reports/phase4_policy/run-local-20261013-v1/application-requests/071b6d6407ff4814ab6a6b46b08404e2/input.png',
          ['Read the main passage','What is the website?','What is the passage about? Could you describe it please?','Read the passage again','What was my third question?']),
      'blurry':('reports/phase4_policy/run-local-20261013-v1/application-requests/d57ffbb579ea495d86fb10c36cc87b93/input.png',
          ['Read the visible text','What is the product name?','Why can\'t you read the text?','How should I take a better picture?']),
      'soup':(sources[4],['What is written on this product label?','What ingredients are listed in the package?',
          'Does the label mention the word sugar?','What is the serving size?','How many calories are there?',
          'What is the net weight?','How do I heat this?','What is the product name?']),
      'living_room':(sources[1],['Could you describe this image?','All objects can you identify in this image?',
          'Is there a person?','Read the visible text','What was my first question?'])}
    # Same living-room image saved by the original permitted application call.
    # Browser-upload caches can expire; never silently substitute another photo.
    durable=Path('reports/phase4_policy/run-local-20261013-v1/application-requests/f04d23e97d0b400e81e15303d92fd8c0/input.png')
    if not Path(fixtures['living_room'][0]).is_file() and durable.is_file():
        fixtures['living_room']=(str(durable),fixtures['living_room'][1])
    records=[];session=FullTextSession('reports/phase4_policy/run-local-20261013-v1')
    previous=json.loads(Path(replay_from).read_text(encoding='utf-8'))['records'] if replay_from else None
    demo,_=build_app(session.run_dir,session)
    for f in demo.fns.values():
        if f.fn and f.fn.__name__=='upload_reset':f.api_visibility='undocumented'
    def save():
        (out/'RESULTS.json').write_text(json.dumps({'records':records,'scope':('Real HTTP/TTS over exact saved manual-fixture OCR/BLIP replay' if previous else 'Five existing manual fixtures; live OCR/BLIP/TTS, not official evaluation'),
          'physical_microphone':'UNVERIFIED','browser_accessibility':'UNVERIFIED','holdout_evaluation':False},indent=2,ensure_ascii=False),encoding='utf-8')
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        _,url,_=demo.queue().launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options())
        for kind,(path,questions) in fixtures.items():
            if not Path(path).is_file():records.append({'fixture':kind,'missing':path});save();continue
            if previous:
                import copy
                examples=[r for r in previous if r['fixture']==kind]
                raw=next((r['details'] for r in examples if r.get('details',{}).get('evidence') is not None and 'orientation' in r['details']),None)
                if raw is None:raw={'evidence':[],'source':{'engine':'RapidOCR'},'orientation':{'selected_ccw_degrees':0},'cache_hit':False,'request_seconds':0.,'ocr_lines':0}
                session.reader.read=lambda *a,_raw=raw,**kw:{'result':copy.deepcopy(_raw),'memory':{}}
                session.easy.read=session.reader.read
                def visual(*args,_examples=examples):
                    r=next(r for r in _examples if r['question']==args[1]);d=r['details'];f=d.get('frozen_vqa_result',d)
                    return d.get('frozen_display_answer',r['answer']),None,'Exact saved visual result replay.',copy.deepcopy(f),None,'Saved evidence.',{'callback_seconds':0.,'model_inference_performed':False}
                session.legacy.request=visual
            client=Client(url,verbose=False);client.predict(handle_file(path),api_name='/upload_reset')
            for n,q in enumerate(questions):
                tick=time.perf_counter()
                try:
                    result=client.predict(handle_file(path),q,n!=0,ASSISTANT,'0',api_name='/analyze')
                    d=next((v for v in result if isinstance(v,dict) and ('final_answer' in v or 'application_error' in v)),{})
                    record={'fixture':kind,'path':str(Path(path).resolve()),'question':q,'answer':result[0],
                        'seconds':time.perf_counter()-tick,'audio':bool(result[1]),'details':d,'history':result[-1],
                        'callback_pass':bool(result[0]) and 'application_error' not in d,
                        'semantic_correctness':'REQUIRES_REVIEW; callback success does not establish correctness'}
                except Exception as e:record={'fixture':kind,'question':q,'exception_type':type(e).__name__,'callback_pass':False,'seconds':time.perf_counter()-tick}
                records.append(record);save();print(kind,q,round(record['seconds'],2),repr(record.get('answer','')[:180]),flush=True)
    finally:demo.close();session.close();save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replay-from');args=p.parse_args();main(args.out,args.replay_from)
