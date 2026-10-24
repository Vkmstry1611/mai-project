"""Opt-in VisionBridge workspace; delegates inference to the existing Session."""
import argparse
import atexit
from pathlib import Path

from workspace_ui import (WorkspaceController,HTML_CSS,BRAND,NAV,HERO,EMPTY_RESPONSE,EMPTY_EVIDENCE,
                          ASSISTANT,VISUAL,READING,MODES,section,status_html,reading_mode)

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets/workspace'


def theme():
    import gradio as gr
    return gr.themes.Base(primary_hue='blue',neutral_hue='slate',font=['Segoe UI','Arial','sans-serif'],font_mono=['Consolas','monospace']).set(
        block_label_background_fill='#f0f1e9',block_label_background_fill_dark='#f0f1e9',
        block_title_text_color='#232820',block_title_text_color_dark='#232820',
        checkbox_background_color='#fbfaf6',checkbox_background_color_dark='#fbfaf6',
        checkbox_background_color_selected='#254ddd',checkbox_background_color_selected_dark='#254ddd',
        checkbox_label_background_fill='transparent',checkbox_label_background_fill_dark='transparent',
        button_secondary_background_fill='#f0f1e9',button_secondary_background_fill_dark='#f0f1e9',
        button_secondary_text_color='#232820',button_secondary_text_color_dark='#232820',
        button_secondary_border_color='#cbd0c1',button_secondary_border_color_dark='#cbd0c1',
        body_background_fill='#f5f4ef',body_background_fill_dark='#f5f4ef',body_text_color='#232820',body_text_color_dark='#232820',
        block_background_fill='#fbfaf6',block_background_fill_dark='#fbfaf6',block_border_color='#d5d8cd',block_border_color_dark='#d5d8cd',
        block_label_text_color='#62675c',block_label_text_color_dark='#62675c',input_background_fill='#fbfaf6',input_background_fill_dark='#fbfaf6',
        # Gradio 6.13 has no input_text_color theme token. Inputs inherit the
        # supported body text color; workspace CSS supplies the explicit color.
        input_border_color='#c9cdbf',input_border_color_dark='#c9cdbf',
        button_primary_background_fill='#254ddd',button_primary_background_fill_dark='#254ddd',button_primary_text_color='#ffffff',button_primary_text_color_dark='#ffffff')


def build_app(run_dir,session=None,model_sessions=None):
    import gradio as gr
    if session is None:
        session=create_session(run_dir)
    controller=WorkspaceController(session)
    if model_sessions is not None:
        from workspace_models import BLIP
    def html(value,**kwargs):return gr.HTML(value,css_template=HTML_CSS,**kwargs)
    with gr.Blocks(title='VisionBridge — visual assistant workspace',analytics_enabled=False,delete_cache=(3600,3600)) as demo:
        prior=gr.State({'workspace_model':model_sessions.default} if model_sessions else None,time_to_live=3600)
        with gr.Column(elem_id='vb-shell',elem_classes=['vb-large-text']):
            html('<nav class="skip-links" aria-label="Skip navigation"><a href="#vb-question">Skip to question</a><a href="#vb-voice">Skip to voice input</a></nav>',elem_id='vb-skip')
            with gr.Row(elem_id='vb-sidebar',equal_height=False):
                html(BRAND)
                html(NAV)
                with gr.Column(elem_id='vb-preferences',scale=0,min_width=190):
                    larger=gr.Checkbox(label='Larger text',value=True,elem_id='vb-access')
                    mute=gr.Checkbox(label='Mute Spoken Answers',value=False,elem_id='vb-mute')
            with gr.Column(scale=1,min_width=320,elem_id='vb-main'):
                html(HERO)
                if model_sessions is not None:
                    with gr.Row(elem_id='vb-model-bar'):
                        model_choice=gr.Radio(model_sessions.choices,value=model_sessions.default,label='Answer model',interactive=True,elem_id='vb-model-choice',scale=1,min_width=260)
                        html('<p class="model-help" style="font-size:16px;line-height:1.7;color:#424c3e"><strong>Choose your perspective.</strong> BLIP gives short visual answers and reads text. Qwen handles broader image questions. Switching clears the response and conversation; your image and draft stay.</p>')
                with gr.Row(elem_id='vb-workspace',equal_height=True):
                    with gr.Column(scale=6,min_width=300,elem_classes='vb-card',elem_id='vb-input-card'):
                        html(section('01','The image','Your starting point. Upload a photo or use your camera.'))
                        image=gr.Image(label='Image — upload or camera',sources=['upload','webcam'],type='pil',height=340,
                            placeholder='Drop an image here, or choose Upload',elem_id='vb-photo')
                        gr.Markdown('Use a clear, upright photo. Avoid glare and keep the text in focus.',elem_classes='vb-note')
                        question=gr.Textbox(label='What would you like to know?',lines=2,max_lines=5,
                            placeholder='What would you like to know about this image?',elem_id='vb-question')
                        if hasattr(session,'for_conversation'):
                            gr.Markdown('You can ask follow-up questions about the same image.',elem_classes='vb-note')
                            gr.Markdown('**Keyboard:** Alt+Q for your question · Alt+V for voice input · Ctrl+Enter to analyze.',elem_classes='vb-note')
                            question_notice=html('<p role="status" aria-live="polite">Type a question, or open voice input below.</p>')
                        with gr.Accordion('Ask with your voice',open=False,elem_classes='vb-inset',elem_id='vb-voice'):
                            gr.Markdown('**1 · Record** → **2 · Stop & review** → **3 · Use question**',elem_classes='vb-note')
                            voice_status=html(status_html('Ready to record','Select Record, speak your question, then select Stop.'),elem_id='vb-voice-status')
                            microphone_hint=gr.Markdown('Allow microphone access when asked. If this browser shows no microphone, upload audio or open this page in Chrome or Edge.',elem_id='vb-microphone-hint')
                            recording=gr.Audio(label='Your recording',sources=['microphone','upload'],type='filepath',format='wav',editable=False,buttons=[],elem_id='vb-recording')
                            speech_transcript=gr.Textbox(label='Your words · edit if needed',placeholder='Your transcript appears here after you stop recording.',interactive=False,lines=2,max_lines=4,elem_id='vb-transcript')
                            with gr.Row(elem_id='vb-voice-actions'):
                                use_voice=gr.Button('Use this question',variant='primary',interactive=False,elem_id='vb-use-voice')
                                transcribe=gr.Button('Retry transcription',interactive=False,elem_id='vb-retry-voice')
                                discard_voice=gr.Button('Start over',elem_id='vb-discard-voice')
                            gr.Markdown('Your typed question stays unchanged until you select Use this question. Prefer a short recording in a quiet room. You can also upload an audio file.',elem_classes='vb-note')
                            with gr.Accordion('Microphone not working?',open=False):
                                microphone_check=gr.Button('Check microphone access',size='sm')
                                microphone_notice=gr.Textbox(label='Microphone help',value='Allow microphone access for this page. If recording is unavailable in the in-app browser, open http://127.0.0.1:7867 in Chrome or Edge, or upload a recording.',interactive=False,lines=3)
                        with gr.Accordion('Advanced / Developer · optional manual overrides',open=False,elem_classes='vb-inset',elem_id='vb-developer',visible=(model_sessions.default==BLIP if model_sessions else not hasattr(session,'backend_label'))) as developer:
                            choices=MODES+(['Full Text Reading (experimental)'] if hasattr(session,'full_request') else [])
                            mode=gr.Radio(choices,value=ASSISTANT,label='Processing override · leave Assistant selected for automatic routing',elem_id='vb-mode')
                            if hasattr(session,'set_engine'):
                                engine=gr.Dropdown(['Automatic','EasyOCR','RapidOCR'],value='Automatic',label='Experimental OCR engine · optional developer override')
                                engine_status=gr.Textbox(label='Engine setting',interactive=False)
                                engine.change(session.set_engine,engine,engine_status,api_name='set_ocr_engine',concurrency_limit=1,concurrency_id='workspace-inference')
                            gr.Markdown('Manual modes are for testing. You do not need to choose a processing method to ask a question.',elem_classes='vb-note')
                            with gr.Group(visible=False,elem_id='vb-rotation',elem_classes='vb-inset') as rotation_group:
                                rotation=gr.Dropdown([('0° · upright','0'),('90° counterclockwise','90'),('180° · upside down','180'),('270° counterclockwise','270')],
                                    value='0',label='Rotate image for manual OCR',info='Only applied with a manual OCR override; automatic reading selects orientation itself.')
                        analyze=gr.Button('Analyze Image',variant='primary',elem_id='vb-analyze')
                        retake=gr.Button('Replace image ↗',size='sm',elem_id='vb-retake')
                        clear=gr.Button('Clear workspace',size='sm',elem_id='vb-clear')
                    with gr.Column(scale=5,min_width=280,elem_classes='vb-card',elem_id='vb-response-card'):
                        html(section('02','The perspective','A closer look at what’s in front of you.'))
                        meta=html(EMPTY_RESPONSE)
                        answer=gr.Textbox(label='Answer / transcription',value='',placeholder='',interactive=False,lines=5,max_lines=16,buttons=['copy'],elem_id='vb-answer')
                        copy=gr.Button('Copy response',size='sm',elem_id='vb-copy')
                        html('<p role="status" aria-live="polite">Copy the answer to keep it handy. You can also select text and press Ctrl+C.</p>',elem_id='vb-copy-status')
                        audio=gr.Audio(label='Play Answer · spoken response',autoplay=True,elem_id='vb-playback')
                        play=gr.Button('Play Answer',size='sm',elem_id='vb-play-answer')
                        gr.Markdown('Speech is generated by default. If your browser blocks autoplay, press Play Answer. Mute Spoken Answers stops playback and disables generation for new requests.',elem_classes='vb-note')
                        status=html(status_html('Ready when you are','Upload a photo, ask a question, then select Analyze Image.'),elem_id='vb-status')
                        gr.Markdown('Answers are generated from your image. Verify important numbers, dates and conclusions.' if hasattr(session,'backend_label') else 'An OCR candidate or learned score is not proof that an answer is correct. Inspect the supporting evidence below.',elem_classes='vb-note')
                        conversation=None
                        if hasattr(session,'for_conversation'):
                            with gr.Accordion('Conversation · recent questions',open=False):
                                conversation=html('<p>No earlier questions for this image.</p>',elem_id='vb-conversation')
                with gr.Accordion('Behind the answer · evidence & details',open=False,elem_classes='vb-card',elem_id='vb-evidence'):
                    html(section('03','A closer look','Inspect the sources, selected regions and original model outputs.'))
                    with gr.Row(elem_id='vb-evidence-panels'):
                        annotated=gr.Image(label='Supporting evidence boxes · all raw and excluded boxes in details',interactive=False,height=285,elem_id='vb-annotated',scale=1,min_width=240)
                        with gr.Column(scale=1,min_width=240):evidence=html(EMPTY_EVIDENCE)
                    with gr.Accordion('Evidence text alternative · complete selection',open=False,elem_classes='vb-inset'):
                        text=gr.Textbox(label='Selected candidates, support IDs and extraction status',interactive=False,lines=7,elem_id='vb-evidence-text')
                    with gr.Accordion('Advanced · decision details & timings',open=False,elem_classes='vb-inset'):
                        gr.Markdown('Original model outputs are preserved. Nested timing measurements overlap; do not add them.',elem_classes='vb-note')
                        routing=html(status_html('Task selection','VisionBridge will choose a task after you ask a question.'),elem_id='vb-routing')
                        details=gr.JSON(label='Original candidate, evidence and decision')
                        timings=gr.JSON(label='Persistent-worker request timings')
                        guidance=gr.JSON(label='Experimental capture quality and before/after comparison · not accuracy')
                capability=getattr(session,'capability_text',None) or ('Explicit reading uses experimental Full Text Reading. Clearly labeled price, brand, expiry and manufacturing fields can be extracted when evidence is sufficient; this does not verify product safety. '
                    if hasattr(session,'full_request') else 'Explicit reading uses the experimental fragment reader. Reliable specific-field extraction is not enabled in this launcher. ')
                html('<section class="guide"><h2>One image.<br>Many possibilities.</h2><p>Upload an image or take a photo, then ask in your own words. You can speak your question, review the transcript, and listen to the answer. Ask a follow-up to explore the same image, or replace it to start a new perspective. '+capability+' Your session is temporary.</p></section>',elem_id='vb-guide')

        outputs=[answer,audio,status,details,annotated,text,timings,meta,analyze,evidence,routing,guidance,prior]
        if conversation is not None:outputs.append(conversation)
        def analyze_browser_values(photo,q,muted,override,angle,previous):
            import time
            callback_start=time.perf_counter()
            from assistive_tools import quality,compare
            from modules.evidence_store import image_identity
            from image_conversation import prepare,history_html,task_quality
            active_session=model_sessions.get((previous or {}).get('workspace_model')) if model_sessions else session
            active_controller=WorkspaceController(active_session) if model_sessions else controller
            state=prepare(previous,photo) if hasattr(active_session,'for_conversation') else previous
            if conversation is not None:
                from uuid import uuid4
                state['submission']={'id':uuid4().hex,'question':str(q or ''),'completed':False,
                    'image_id':state['conversation']['image_id']}
            for result in active_controller.analyze_ui(photo,q,muted,override,angle,context=state):
                info={};values=list(result)
                if conversation is not None and result[0] and not result[3].get('application_error'):
                    state['submission']['completed']=True;state['last_submitted_question']=str(q or '')
                if photo is not None and result[0]:
                    task=result[3].get('assistant_trace',{}).get('route','reading' if result[6].get('experimental_reading_active') else 'vqa')
                    metrics=(result[3].get('quality') or task_quality(photo,task,result[3].get('evidence',[]))) if hasattr(session,'for_conversation') else quality(photo)
                    state=dict(state or {},image_id=image_identity(photo),quality=metrics,ocr_lines=result[3].get('ocr_lines',0))
                    local=result[3].get('local_quality',{})
                    info={'quality':metrics,'local_text_quality':local,'comparison':compare(previous,state),'retake':'Replace the image above, keep the question and Analyze again. No correctness guarantee.'}
                    suggestions=' '.join(metrics['warnings']+local.get('warnings',[]))
                    if suggestions:values[2]+=status_html('Experimental capture guidance',suggestions+' These are heuristic suggestions; inference was not blocked.')
                values[6]=dict(values[6],workspace_generator_wall_seconds=time.perf_counter()-callback_start)
                if 'assistant_trace' in values[3]:
                    values[3]=dict(values[3],assistant_trace=dict(values[3]['assistant_trace'],total_api_generator_seconds=values[6]['workspace_generator_wall_seconds']))
                yield (*values,info,state,*([history_html(state)] if conversation is not None else []))
        def analyze_browser(photo,q,muted,override,angle,previous):
            # Catch UI/state/routing failures outside the controller as well.
            # This is an application error, never a fabricated model abstention.
            from workspace_diagnostics import report_failure
            latest=None
            try:
                for value in analyze_browser_values(photo,q,muted,override,angle,previous):
                    latest=value
                    yield value
            except Exception as error:
                reference=report_failure('workspace-callback',error)
                message=status_html('Application error','The request could not finish. No substitute answer was generated. Diagnostic reference: '+reference,'error')
                if latest is not None and latest[0]:
                    values=list(latest);values[2]+=message;values[8]=gr.update(interactive=True)
                    yield tuple(values)
                else:
                    from image_conversation import history_html
                    kept=previous if isinstance(previous,dict) else None
                    try:history=history_html(kept)
                    except Exception:history='<p>Earlier context could not be displayed. You can clear the session and retry.</p>'
                    yield ('',None,message,{'application_error':{'stage':'workspace-callback','reference':reference}},None,'',{},
                        EMPTY_RESPONSE,gr.update(interactive=True),EMPTY_EVIDENCE,
                        status_html('Task status','Please retry. If this continues, restart the local application and share the diagnostic reference.'),{},kept,
                        *([history] if conversation is not None else []))
        event=analyze.click(analyze_browser,[image,question,mute,mode,rotation,prior],outputs,api_name='analyze',show_progress='minimal',concurrency_limit=1,concurrency_id='workspace-inference',trigger_mode='once')
        if conversation is not None:
            def clear_submitted_question(current,state):
                from workspace_turns import question_after_submission
                value=question_after_submission(current,state)
                return (gr.update(value='') if value=='' else gr.skip(),
                    '<p role="status" aria-live="polite">Response complete. Enter a new question or a follow-up; the previous question is in Recent questions.</p>' if value=='' else gr.skip())
            # Clear when the first text answer arrives, rather than making a
            # user wait for potentially slow speech synthesis before a follow-up.
            answer.change(clear_submitted_question,[question,prior],[question,question_notice],queue=False,
                api_name='clear_submitted_question',api_visibility='undocumented')
            event.then(clear_submitted_question,[question,prior],[question,question_notice],queue=False,
                api_name='clear_submitted_after_audio',api_visibility='private')
        demo.load(None,outputs=mute,js="() => {if(!window.vbSpeechGuard){document.addEventListener('play',(e)=>{if(e.target.closest?.('#vb-playback')){try{if(sessionStorage.getItem('visionbridge-mute')==='true'){e.target.muted=true;e.target.pause()}}catch(x){}}},true);window.vbSpeechGuard=true}try{return sessionStorage.getItem('visionbridge-mute')==='true'}catch(e){return false}}",api_visibility='private')
        mute.change(None,mute,[],js="(muted) => {try{sessionStorage.setItem('visionbridge-mute',String(muted))}catch(e){} const a=document.querySelector('#vb-playback audio'); if(a){a.muted=muted;if(muted)a.pause()} return [];}",queue=False,api_visibility='private')
        play.click(None,inputs=mute,outputs=[],js="(muted) => {const a=document.querySelector('#vb-playback audio');if(a&&!muted){a.muted=false;a.play().catch(()=>{});}return [];}",queue=False,api_visibility='private')
        copy.click(None,inputs=answer,outputs=[],js="async (text) => {try{await navigator.clipboard.writeText(text)}catch(e){document.querySelector('#vb-answer textarea')?.focus()}return [];}",queue=False,api_visibility='private')
        def reset(photo=None):
            return ('',None,status_html('Image ready' if photo is not None else 'Ready when you are',
                'Ask a question, then select Analyze Image.' if photo is not None else 'Upload a photo to begin.'),{},None,'',{},EMPTY_RESPONSE,gr.update(interactive=True),EMPTY_EVIDENCE,
                status_html('Task selection','No task has run for this input.'))
        original_reset=reset
        def reset(photo=None):return (*original_reset(photo),{},gr.skip(),*(['<p>Context will reset for a changed image on the next question.</p>'] if conversation is not None else []))
        if conversation is not None:
            def upload_reset(photo,previous):
                from image_conversation import prepare,history_html
                try:
                    fresh=prepare(previous,photo)
                    return (*original_reset(photo),{},fresh,history_html(fresh))
                except Exception as error:
                    from workspace_diagnostics import report_failure
                    reference=report_failure('image-upload-state',error)
                    values=list(original_reset(photo))
                    values[2]=status_html('Session state error','Conversation context was cleared. Try Analyze again. Diagnostic reference: '+reference,'error')
                    values[3]={'application_error':{'stage':'image-upload-state','reference':reference}}
                    return (*values,{},None,'<p>Conversation context cleared after a state error.</p>')
            image.change(upload_reset,[image,prior],outputs,cancels=[event],queue=False,api_visibility='private')
        else:image.change(reset,image,outputs,cancels=[event],queue=False,api_visibility='private')
        def reuse_last_question(current,state):
            from workspace_turns import question_for_retake
            return question_for_retake(current,state)
        retake.click(reuse_last_question,[question,prior],question,queue=False,api_name='reuse_last_question',api_visibility='undocumented').then(None,[],[],
            js="() => {const el=document.querySelector('#vb-photo');el?.scrollIntoView({block:'center'});el?.querySelector('input')?.focus();return [];}",queue=False,api_visibility='private')
        if model_sessions is not None:
            def select_model(value):
                if value not in model_sessions.choices:raise ValueError('Select a supported model.')
                values=list(original_reset())
                values[2]=status_html(value+' selected','Your image and draft are kept. Ask a new question to use this model.')
                return (*values,{}, {'workspace_model':value},'<p>No earlier questions for this model.</p>',
                    gr.update(value=ASSISTANT),gr.update(visible=False),gr.update(visible=value==BLIP))
            model_choice.change(select_model,model_choice,[*outputs,mode,rotation_group,developer],cancels=[event],
                queue=True,concurrency_limit=1,concurrency_id='workspace-inference',api_name='select_model')
            def clear_selected(value):
                return (None,'',*original_reset(),{}, {'workspace_model':value},'<p>Image and conversation context cleared.</p>')
            clear.click(clear_selected,model_choice,[image,question,*outputs],cancels=[event],queue=False,api_name='reset_session')
        else:
            clear.click(lambda:(None,'',*original_reset(),{},None,*(['<p>Image and conversation context cleared.</p>'] if conversation is not None else [])),outputs=[image,question,*outputs],cancels=[event],queue=False,api_name='reset_session')
        def switch_mode(value):
            reading=value=='Full Text Reading (experimental)' or reading_mode(value)
            return (gr.update(visible=reading),gr.update(placeholder='What words are written in this image?' if reading else 'What would you like to know about this image?'),*reset())
        mode.change(switch_mode,mode,[rotation_group,question,*outputs],cancels=[event],queue=False,api_name='set_mode')
        from workspace_voice import bind as bind_voice,MICROPHONE_CHECK_JS,MICROPHONE_AVAILABILITY_JS
        bind_voice(recording,transcribe,question,status,controller,gr,speech_transcript,clear,voice_status,use_voice,discard_voice)
        demo.load(None,outputs=microphone_hint,js=MICROPHONE_AVAILABILITY_JS,api_visibility='private')
        microphone_check.click(None,inputs=[],outputs=microphone_notice,js=MICROPHONE_CHECK_JS,queue=False,api_visibility='private')
        demo.load(None,outputs=[],api_visibility='private',js="""() => {
          const question=document.querySelector('#vb-question textarea');
          question?.setAttribute('aria-keyshortcuts','Alt+Q Control+Enter');
          if(!window.vbKeyboardAccess){
            document.addEventListener('keydown',(e)=>{
              if(e.altKey&&!e.ctrlKey&&e.key.toLowerCase()==='q'){
                e.preventDefault();const q=document.querySelector('#vb-question textarea');q?.focus();q?.scrollIntoView({block:'center'});
              }else if(e.altKey&&!e.ctrlKey&&e.key.toLowerCase()==='v'){
                e.preventDefault();const b=document.querySelector('#vb-voice>button');b?.focus();b?.scrollIntoView({block:'center'});
              }else if(e.ctrlKey&&e.key==='Enter'&&document.activeElement?.closest('#vb-question')){
                e.preventDefault();document.querySelector('#vb-analyze:not(:disabled)')?.click();
              }
            });window.vbKeyboardAccess=true;
          }return [];
        }""")
        larger.change(None,larger,[],js="(large) => {document.getElementById('vb-shell')?.classList.toggle('vb-large-text', large); return [];}",queue=False,api_visibility='private')
    atexit.register(model_sessions.close if model_sessions else session.close)
    return demo,session


def launch_options():
    return dict(server_name='127.0.0.1',share=False,show_error=True,theme=theme(),css=(ASSETS/'workspace.css').read_text(encoding='utf-8'),
                allowed_paths=[str(ASSETS)])


def create_session(run_dir,backend='auto',model=None):
    required=[Path(run_dir)/'POLICY_FREEZE.json',ROOT/'models/blip_vizwiz_v2/config.json',
              ROOT/'.venv-gpu/Scripts/python.exe']
    if backend=='local' or (backend=='auto' and not all(p.is_file() for p in required)):
        from modules.local_vision import LocalVisionSession
        return LocalVisionSession(model)
    if not all(p.is_file() for p in required):
        raise RuntimeError('Frozen backend is incomplete. Restore the private runtime bundle or use --backend local.')
    from full_text_session import FullTextSession
    return FullTextSession(run_dir)


if __name__=='__main__':
    import sys
    import os
    # Keep MAI uploads independent of BDA Gradio cache cleanup.
    os.environ.setdefault('GRADIO_TEMP_DIR',str(ROOT/'outputs/gradio-cache'))
    if Path(sys.prefix).name!='.venv':raise RuntimeError('Use .venv for Gradio; the persistent worker uses .venv-gpu')
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='reports/phase4_policy/run-local-20261013-v1');p.add_argument('--port',type=int,default=7865)
    p.add_argument('--backend',choices=['auto','local','frozen'],default='auto')
    p.add_argument('--vision-model',default=None);args=p.parse_args()
    session=create_session(Path(args.run_dir).resolve(),args.backend,args.vision_model)
    from workspace_models import FrontendModels,BLIP,QWEN
    default_model=QWEN if hasattr(session,'backend_label') else BLIP
    models=FrontendModels(lambda name:create_session(Path(args.run_dir).resolve(),'frozen' if name==BLIP else 'local',args.vision_model),default_model,session)
    # Build both sets of existing controls; model weights still load on demand.
    session=models.get(BLIP)
    print('Starting VisionBridge: frontend model selector; default='+default_model,flush=True)
    demo,session=build_app(Path(args.run_dir).resolve(),session=session,model_sessions=models)
    try:demo.queue(default_concurrency_limit=1).launch(server_port=args.port,**launch_options())
    finally:models.close()
