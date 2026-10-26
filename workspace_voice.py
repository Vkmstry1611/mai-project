"""Workspace-only validated, draft-safe speech review. Raw audio is never copied."""
import hashlib
import time
from pathlib import Path
from collections import OrderedDict

# Explicit user action only: permission/device probe; stop all tracks immediately.
# No recording, upload or Whisper inference is performed by this diagnostic.
MICROPHONE_CHECK_JS="""async () => {
 const notice = (text) => [text];
 if(!window.isSecureContext || !navigator.mediaDevices?.getUserMedia)
   return notice('Microphone capture is unavailable here. Open the localhost workspace in Chrome, or upload a WAV recording.');
 let stream, timer, expired=false;
 try {
   const pending=navigator.mediaDevices.getUserMedia({audio:true,video:false});
   pending.then(s=>{if(expired)s.getTracks().forEach(track=>track.stop())}).catch(()=>{});
   stream=await Promise.race([pending,new Promise((_,reject)=>{timer=setTimeout(()=>{expired=true;const e=new Error();e.name='TimeoutError';reject(e)},10000)})]);
   return notice('Microphone access granted. No recording was saved. Choose Record, speak, then Stop to test transcription.');
 } catch(e) {
   const messages={NotAllowedError:'Microphone permission was denied. Allow Microphone in the site controls beside the address, check Windows microphone privacy settings, and try again.',
    NotFoundError:'No microphone input device was found. Connect or select a microphone, or upload a WAV recording.',
    NotReadableError:'The microphone could not be opened. Close other recording applications and check the selected Windows input device.',
    SecurityError:'This browser blocked microphone capture. Use the localhost workspace in Chrome and check site permissions.',
    TimeoutError:'No microphone access response arrived. Check the browser permission prompt, or open the localhost workspace in Chrome. You can upload a WAV recording instead.'};
   return notice(messages[e.name] || 'Microphone access failed. Check browser permissions and your input device, or upload a WAV recording.');
 } finally {clearTimeout(timer);stream?.getTracks().forEach(track=>track.stop());}
}"""


# Device enumeration does not request permission or start recording.
MICROPHONE_AVAILABILITY_JS="""async () => {
 try {
   if(navigator.mediaDevices?.enumerateDevices){
     const devices=await Promise.race([navigator.mediaDevices.enumerateDevices(),new Promise(resolve=>setTimeout(()=>resolve([]),2000))]);
     if(devices.some(d=>d.kind==='audioinput'))return ['Allow microphone access when asked. Select Record, speak, then Stop.'];
   }
 } catch(e) {}
 return ['This browser has not made a microphone available. Upload an audio file, or open this page in Chrome or Edge and allow microphone access.'];
}"""


def transcribe_local(recording):
    # Keep CTranslate2's native runtime out of the Gradio process on Windows.
    # Loading both in one process can terminate Python without an exception.
    import json
    import subprocess
    import sys
    import os
    root=Path(__file__).resolve().parent
    result=subprocess.run([sys.executable,str(root/'workspace_speech_worker.py'),str(Path(recording).resolve())],
        cwd=root,capture_output=True,text=True,encoding='utf-8',timeout=240,
        env=dict(os.environ,PYTHONIOENCODING='utf-8'),
        creationflags=0x08000000 if sys.platform=='win32' else 0)
    if result.returncode:
        raise RuntimeError('Local speech worker failed. Check the Whisper model installation and retry.')
    payload=json.loads(result.stdout)
    if not payload.get('text'):raise ValueError('No speech was recognized. Try again or type your question.')
    return payload['text'],'Transcribed; review and edit your question.'


def validate_audio(path):
    import wave
    import audioop
    p=Path(path)
    if not p.is_file():raise ValueError('No captured audio file is available. Record again or upload a recording.')
    if p.stat().st_size>25*1024*1024:raise ValueError('Recording is too large. Please record a short question.')
    try:
        with wave.open(str(p),'rb') as f:
            if f.getnframes()==0 or f.getframerate()<8000 or f.getnchannels() not in (1,2):raise ValueError('Empty or unsupported audio. Record a short question again.')
            if f.getnframes()/f.getframerate()>120:raise ValueError('Please keep spoken questions under two minutes.')
            width=f.getsampwidth();data=f.readframes(f.getnframes())
            if width==1:data=audioop.bias(data,1,-128)
            if not data or audioop.rms(data,width)/float(2**(8*width-1))<.0001:raise ValueError('No audible signal was captured. Check microphone permission, selected input and recording level, then record again.')
    except ValueError:raise
    except Exception as error:raise ValueError('The audio format could not be decoded. Record again or upload a WAV recording.') from error
    return hashlib.sha256(p.read_bytes()).hexdigest()


def insert_transcript(current,snapshot,text):
    """Run after transcription, with the browser's latest question value."""
    if not text:return None,'No transcript was inserted. You can type your question.'
    if str(current or '')!=str(snapshot or ''):
        return None,'Your newer typed question was preserved. Record again if you want to replace it.'
    return text,'Question captured. Review it, then select Get Answer.'


class VoiceReview:
    def __init__(self,transcriber):self.transcriber=transcriber;self.recent=OrderedDict()

    def transcribe(self,recording,state=None,session_id=None):
        state=dict(state or {});start=time.perf_counter()
        if not recording:return '', 'No recording was captured. Allow microphone access for localhost, stop recording, or upload audio. You can also type your question.',state
        try:
            key=validate_audio(recording)
            memo=(session_id,key)
            if session_id and memo in self.recent and time.monotonic()-self.recent[memo][0]<900:
                state=dict(self.recent[memo][1])
            if state.get('audio_sha256')==key and state.get('text'):
                return state['text'],'Transcription ready (reused; no duplicate Whisper inference).',state
            text,status=self.transcriber(recording)
            if not str(text or '').strip():return '', 'Speech recognition returned no text. Try a clearer recording or type your question.',state
            state={'audio_sha256':key,'text':str(text).strip(),'seconds':time.perf_counter()-start}
            if session_id:
                self.recent[memo]=(time.monotonic(),dict(state))
                while len(self.recent)>16:self.recent.popitem(last=False)
            return state['text'],'Transcription ready. Review the text before submitting.',state
        except ValueError as error:return '',str(error),state
        except Exception as error:
            from workspace_diagnostics import report_failure
            ref=report_failure('microphone-transcription',error)
            return '', 'Speech processing failed. You can type your question. Diagnostic reference: '+ref,state


def voice_job(audio=None,draft='',recording=False):
    """A new recording/reset invalidates every older completion."""
    from uuid import uuid4
    return {'id':uuid4().hex,'audio':audio,'draft':str(draft or ''),'recording':recording}


def current_completion(job,result):
    return bool(job and result and job.get('id')==result.get('id'))


def bind(recording,button,question,status,controller,gr,transcript,clear_button,voice_status,use_button,discard_button,record_toggle=None):
    from workspace_ui import status_html
    voice=VoiceReview(controller.transcriber)
    cache=gr.State(None,time_to_live=3600)
    job=gr.State(None,time_to_live=3600)
    completion=gr.State(None,time_to_live=3600)
    jobs=[]
    local_outputs=[job,transcript,voice_status,use_button,button]

    def begin(audio,current_question='',previous_job=None):
        draft=(previous_job or {}).get('draft',current_question) if (previous_job or {}).get('recording') else current_question
        return (voice_job(audio,draft),gr.update(value='',interactive=False),
            (status_html('Transcribing your recording','This usually takes a few seconds. You can keep typing your question.','working') if audio else status_html('Ready for your recording','Select Record, or upload an audio file.')),
            gr.update(interactive=False),gr.update(interactive=False))

    def run(snapshot,previous,request=None):
        if not (snapshot or {}).get('audio'):return {'id':(snapshot or {}).get('id'),'pending':True}
        text,message,new=voice.transcribe((snapshot or {}).get('audio'),previous,getattr(request,'session_hash',None))
        return {'id':(snapshot or {}).get('id'),'text':text,'message':message,'cache':new}
    run.__annotations__={'request':gr.Request}

    def finish(current,result):
        # Clear/re-record may happen while the isolated worker is still running.
        if not current_completion(current,result) or result.get('pending'):return (gr.skip(),)*5
        text=result['text']
        return (gr.update(value=text,interactive=bool(text)),
            status_html('Speech recognized' if text else 'Try recording again',
                'Adding your words to the question box.' if text else result['message'],
                'done' if text else 'error'),result['cache'],
            gr.update(interactive=bool(text)),gr.update(interactive=bool(current.get('audio'))))

    def auto_insert(current,result,current_question):
        if not current_completion(current,result) or result.get('pending') or not result.get('text'):
            return gr.skip(),gr.skip()
        value,message=insert_transcript(current_question,current.get('draft'),result['text'])
        return (value if value is not None else gr.skip()),status_html('Question ready' if value is not None else 'Typed question kept',message,'done')

    def attach(event):
        work=event.then(run,[job,cache],completion,show_progress='hidden',concurrency_limit=1,
            concurrency_id='workspace-speech-input',api_visibility='private')
        work.success(finish,[job,completion],[transcript,voice_status,cache,use_button,button],queue=False,api_visibility='private').then(
            auto_insert,[job,completion,question],[question,voice_status],queue=False,api_visibility='private')
        jobs.append(work)

    # Only the completed file triggers transcription. Stop never starts another job.
    attach(recording.input(begin,[recording,question,job],local_outputs,queue=False,api_visibility='private'))
    attach(button.click(begin,[recording,question,job],local_outputs,queue=False,trigger_mode='once',api_visibility='private'))

    def use_question(text):
        if not str(text or '').strip():
            return gr.skip(),status_html('No question yet','Record a question first.','error')
        return str(text).strip(),status_html('Question ready','Your question is in the question box. Select Get Answer when you are ready.','done')
    use_button.click(use_question,transcript,[question,voice_status],queue=False,api_name='use_voice_question').then(
        None,outputs=[],queue=False,api_visibility='private',
        js="() => {const q=document.querySelector('#vb-question textarea');q?.focus();q?.scrollIntoView({block:'center'});return [];}")

    def starting(current_question=''):
        return (voice_job(draft=current_question,recording=True),gr.update(value='',interactive=False),
            status_html('Recording…','Speak naturally. Select Stop when you finish.','working'),
            gr.update(interactive=False),gr.update(interactive=False))
    recording.start_recording(starting,inputs=question,outputs=local_outputs,queue=False,api_visibility='private',
        js="() => {document.querySelector('#vb-playback audio')?.pause(); return []}")
    recording.stop_recording(lambda:None,outputs=[],queue=False,api_visibility='private')
    if record_toggle is not None:
        recording.start_recording(lambda:gr.update(value='Stop recording'),outputs=record_toggle,queue=False,api_visibility='private')
        recording.stop_recording(lambda:gr.update(value='Record question'),outputs=record_toggle,queue=False,api_visibility='private')
        recording.clear(lambda:gr.update(value='Record question'),outputs=record_toggle,queue=False,api_visibility='private')
        clear_button.click(lambda:gr.update(value='Record question'),outputs=record_toggle,queue=False,api_visibility='private')

    def reset_voice(request=None):
        session_id=getattr(request,'session_hash',None)
        for key in list(voice.recent):
            if key[0]==session_id:voice.recent.pop(key,None)
        return (voice_job(),gr.update(value='',interactive=False),
            status_html('Voice input','Select Record question, then Stop recording.'),
            gr.update(interactive=False),gr.update(interactive=False),None)
    reset_voice.__annotations__={'request':gr.Request}
    recording.clear(reset_voice,outputs=[*local_outputs,cache],queue=False,api_visibility='private',cancels=jobs)
    def discard(request=None):return (gr.update(value=None,recording=False),*reset_voice(request))
    discard.__annotations__={'request':gr.Request}
    for control in (discard_button,clear_button):
        control.click(discard,outputs=[recording,*local_outputs,cache],queue=False,api_visibility='private',cancels=jobs)

    # Keep existing speech APIs for integrations; the simple UI inserts completed speech automatically.
    legacy=gr.Button(visible=False)
    legacy.click(controller.transcribe,recording,[transcript,status],api_name='transcribe_question',api_visibility='undocumented')
    def checked(audio,previous,request=None):return voice.transcribe(audio,previous,getattr(request,'session_hash',None))
    checked.__annotations__={'request':gr.Request}
    api=gr.Button(visible=False)
    api.click(checked,[recording,cache],[transcript,status,cache],api_name='transcribe_validated',api_visibility='undocumented',concurrency_id='workspace-speech-input',concurrency_limit=1)
    def apply(current,snapshot,text):
        value,message=insert_transcript(current,snapshot,text)
        return (value if value is not None else gr.skip()),status_html('Review your question',message,'done')
    compatibility=gr.Button(visible=False)
    draft=gr.State('',time_to_live=3600)
    compatibility.click(apply,[question,draft,transcript],[question,status],queue=False,
        api_name='insert_reviewed_transcript',api_visibility='undocumented')
