"""Real localhost API smoke tests. Browser rendering/physical input unverified."""
import json
import os
from pathlib import Path
import socket
import time
import uuid
import wave


def main(notices_only=False):
    os.environ['GRADIO_ANALYTICS_ENABLED']='False'
    os.environ['HF_HUB_OFFLINE']='1'
    out=Path('reports/workspace_ui')/('validation-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    from PIL import Image,ImageDraw,ImageFont
    from visionbridge.launch_persistent_demo import Session
    from launch_workspace import build_app,launch_options
    from visionbridge.workspace_ui import ASSISTANT,VISUAL,READING
    from gradio_client import Client,handle_file
    from urllib.request import urlopen
    image=Image.new('RGB',(600,250),'white')
    ImageDraw.Draw(image).text((70,70),'HELLO',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',90),fill='black')
    fixture=out/'synthetic-hello.png';image.save(fixture)
    real=Session('reports/phase4_policy/run-local-20261013-v1')
    class Observed:
        def __init__(self):self.calls=[];self.latest=None
        def request(self,*args):
            self.calls.append(dict(question=args[1],speak=args[2],reading=args[3],rotation=args[4]))
            self.latest=real.request(*args);return self.latest
        def close(self):real.close()
    observed=Observed();demo,_=build_app(real.run_dir,session=observed)
    records=[];checks={}
    def save(name,data):
        with (out/(name+'.json')).open('x',encoding='utf-8') as file:json.dump(data,file,indent=2,ensure_ascii=False)
    def check(name,condition):
        checks[name]=bool(condition)
        save('checks-progress-'+str(len(checks)),checks)
        if not condition:raise AssertionError(name)
    def analyze(name,mode,rotation,speak,question='What words are written in this image?'):
        tick=time.perf_counter()
        result=client.predict(handle_file(str(fixture)),question,not speak,mode,rotation,api_name='/analyze')
        # gradio_client omits components such as Button with skip_api=True.
        details=next(value for value in result if isinstance(value,dict) and 'final_answer' in value)
        timings=next(value for value in result if isinstance(value,dict) and 'persistent_worker' in value)
        record=dict(name=name,seconds=time.perf_counter()-tick,result=result,details=details,timings=timings)
        records.append(record);save(name,record)
        check(name+'_original_answer_preserved',result[0]==observed.latest[0])
        check(name+'_original_details_preserved',details==observed.latest[3])
        return result,details,timings
    try:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options())
        with urlopen(url+'config') as response:config=json.load(response)
        save('served-config',config)
        check('server_started',config['title']=='VisionBridge — visual assistant workspace')
        client=Client(url,verbose=False)
        api=client.view_api(return_format='dict',print_info=False);save('api-schema',api)
        check('public_endpoints',all(name in api['named_endpoints'] for name in ('/analyze','/set_mode','/transcribe_question')))
        print('Server started:',url,flush=True)
        before=len(observed.calls)
        mode_result=client.predict(READING,api_name='/set_mode');save('mode-reading-api',{'result':mode_result})
        client.predict(VISUAL,api_name='/set_mode')
        check('mode_switch_does_not_infer',len(observed.calls)==before)
        invalid=client.predict(None,'Read this',False,VISUAL,'0',api_name='/analyze')
        check('missing_image_rejected_without_inference',len(observed.calls)==before and 'Check your input' in str(invalid))
        missing=client.predict(None,api_name='/transcribe_question')
        check('missing_recording_rejected','Record a question first' in str(missing))

        if notices_only:
            for index,(question,task) in enumerate([('Is this expired?','unsupported_field'),('Tell me what this costs.','unsupported_field'),
                ('Can you help me?','clarify'),('Transcribe the sign.','clarify')]):
                notice=client.predict(handle_file(str(fixture)),question,False,ASSISTANT,'0',api_name='/analyze')
                check('notice_case_'+str(index),any(isinstance(v,dict) and v.get('decision')=='workspace_'+task for v in notice))
            check('notices_start_no_worker',real.client.process is None and not observed.calls)
            save('RESULTS',dict(checks=checks,scope='Supplemental notice-only API checks; no model inference',
                browser_interactions='UNVERIFIED',holdout_inference=False,training=False))
            print('Passed notice-only API checks:',sum(v is True for v in checks.values()),'; saved:',out,flush=True)
            return

        print('Running real OCR and TTS on synthetic HELLO',flush=True)
        result,details,timings=analyze('reading-with-audio',READING,'0',True)
        check('real_hello_preserved',result[0]=='Recognized text fragments: HELLO.')
        check('evidence_panel_uses_real_ocr',any('evidence-list' in str(value) and 'HELLO' in str(value) for value in result) and bool(details['supporting_evidence_ids']))
        with wave.open(str(result[1])) as audio:check('answer_audio_nonempty',audio.getnframes()>0)
        result,details,timings=analyze('reading-cached-repeat',READING,'0',False)
        check('ocr_cache_reused',details['extraction_status']['ocr']['cache_hit'])
        check('speech_off_has_no_audio',result[1] is None)

        print('Running real frozen-policy request; hidden rotation must be ignored',flush=True)
        result,details,timings=analyze('visual-qa',VISUAL,'180',False)
        check('visual_qa_route_preserved',not timings['experimental_reading_active'] and timings['rotation_ccw_degrees']==0)
        check('cuda_blip_used',details.get('gpu_runtime',{}).get('device')=='NVIDIA GeForce GTX 1650' and bool(details.get('baseline_answer')))

        def comparable(value):
            value=json.loads(json.dumps(value))
            for state in value.get('extraction_status',{}).values():state.pop('cache_hit',None)
            return value
        print('Testing automatic routing and parity with manual overrides',flush=True)
        automatic,auto_details,auto_time=analyze('automatic-reading',ASSISTANT,'180',False)
        manual=next(r for r in records if r['name']=='reading-cached-repeat')
        check('automatic_reader_selected',auto_time['experimental_reading_active'] and auto_time['rotation_ccw_degrees']==0)
        check('automatic_manual_reading_parity',automatic[0]==manual['result'][0] and comparable(auto_details)==comparable(manual['details']))
        check('routing_explained_in_technical_panel','automatic' in str(automatic[-1]) and 'reading' in str(automatic[-1]))
        _,_,alias_time=analyze('automatic-label-says',ASSISTANT,'0',False,'Tell me what the label says.')
        check('natural_label_reading_phrase_supported',alias_time['experimental_reading_active'])
        color_question='What color is the background?'
        automatic,auto_details,auto_time=analyze('automatic-visual',ASSISTANT,'180',False,color_question)
        manual,manual_details,_=analyze('manual-visual-comparison',VISUAL,'0',False,color_question)
        check('automatic_frozen_vqa_selected',not auto_time['experimental_reading_active'] and auto_time['rotation_ccw_degrees']==0)
        check('automatic_manual_frozen_parity',automatic[0]==manual[0] and comparable(auto_details)==comparable(manual_details))
        before=len(observed.calls)
        notice=client.predict(handle_file(str(fixture)),'What is the expiry date?',False,ASSISTANT,'0',api_name='/analyze')
        notice_details=next(v for v in notice if isinstance(v,dict) and v.get('decision')=='workspace_unsupported_field')
        check('specific_field_notice_without_inference',len(observed.calls)==before and 'cannot reliably' in notice[0] and not notice_details['selected_evidence'])
        with wave.open(str(notice[1])) as audio:check('limitation_notice_spoken_audio_generated',audio.getnframes()>0)
        clarification=client.predict(handle_file(str(fixture)),'Help',False,ASSISTANT,'0',api_name='/analyze')
        check('ambiguous_request_clarifies_without_inference',len(observed.calls)==before and 'one specific question' in clarification[0])
        save('assistant-notices',{'specific_field':notice,'clarification':clarification})

        from config import Config
        cache=Config().cache_dir/'models--Systran--faster-whisper-tiny.en'
        cached=any(cache.glob('snapshots/*/model.bin'))
        if cached:
            from dataclasses import replace
            from modules.text_to_speech import synthesize
            speech=synthesize('What words are written in this image?',replace(Config(),output_dir=out/'speech-input'))
            print('Testing cached Faster Whisper with synthetic spoken question',flush=True)
            transcript=client.predict(handle_file(speech),api_name='/transcribe_question')
            save('real-transcription',{'result':transcript,'input_audio':speech})
            check('real_transcription_and_review',bool(transcript[0].strip()) and 'Review your question' in str(transcript))
        else:checks['real_transcription_and_review']='UNVERIFIED: cached Whisper weights missing; no download attempted'
        save('RESULTS',dict(checks=checks,records=records,observed_requests=observed.calls,
            browser_interactions='UNVERIFIED: Computer Use previously stopped; no browser automation attempted',
            microphone_physical_recording='UNVERIFIED',audible_playback='UNVERIFIED',larger_text_js_execution='UNVERIFIED',
            server_url=url,holdout_inference=False,training=False))
        print('Passed API/backend checks:',sum(v is True for v in checks.values()),'; saved:',out,flush=True)
    finally:demo.close();observed.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--notices-only',action='store_true')
    main(parser.parse_args().notices_only)
