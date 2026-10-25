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


def transcribe_local(recording):
    from modules import speech_to_text
    from config import Config
    return speech_to_text.transcribe(recording,Config()),'Transcribed; review and edit your question.'


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
        return None,'Transcription ready below. Your newer typed draft was preserved; copy or edit the transcript if needed.'
    return text,'Transcription ready. Review and edit the question, then select Analyze Image.'


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


def bind(recording,button,question,status,controller,gr,transcript,clear_button):
    from workspace_ui import status_html
    voice=VoiceReview(controller.transcriber)
    draft=gr.State('',time_to_live=3600)
    cache=gr.State(None,time_to_live=3600)
    jobs=[]
    def capture(current):return str(current or '')
    def run(audio,previous,request=None):
        yield gr.skip(),status_html('Processing speech','Please wait. Your recording is processed locally.','working'),gr.skip()
        text,message,new=voice.transcribe(audio,previous,getattr(request,'session_hash',None))
        yield text,status_html('Transcription ready' if text else 'Speech input error',message,'done' if text else 'error'),new
    run.__annotations__={'request':gr.Request}
    def apply(current,snapshot,text):
        if not text:return gr.skip(),gr.skip()  # Preserve the specific capture/decoder error from run.
        value,message=insert_transcript(current,snapshot,text)
        return (value if value is not None else gr.skip()),status_html('Review your question' if text else 'Speech input error',message,'done' if text else 'error')
    def attach(event):
        job=event.then(run,[recording,cache],[transcript,status,cache],show_progress='minimal',concurrency_limit=1,concurrency_id='workspace-speech-input',api_name=False)
        insertion=job.success(apply,[question,draft,transcript],[question,status],queue=False,api_name='insert_reviewed_transcript',api_visibility='undocumented')
        jobs.append(job)  # Gradio may cancel queued jobs, not queue=False insertions.
    # input fires when the completed recording/upload value arrives, rather
    # than stop_recording, which can precede file conversion/upload completion.
    attach(recording.input(capture,question,draft,queue=False,api_visibility='private'))
    attach(button.click(capture,question,draft,queue=False,trigger_mode='once',api_visibility='private'))
    # Backwards-compatible one-audio/two-output endpoint, not a second UI trigger.
    legacy=gr.Button(visible=False)
    legacy.click(controller.transcribe,recording,[transcript,status],api_name='transcribe_question',api_visibility='undocumented')
    def checked(audio,previous,request=None):return voice.transcribe(audio,previous,getattr(request,'session_hash',None))
    checked.__annotations__={'request':gr.Request}
    api=gr.Button(visible=False)
    api.click(checked,[recording,cache],[transcript,status,cache],api_name='transcribe_validated',api_visibility='undocumented',concurrency_id='workspace-speech-input',concurrency_limit=1)
    def dashboard_transcribe(audio):
        import subprocess
        import sys
        validate_audio(audio)
        script=("from modules.speech_to_text import transcribe; "
                "from config import Config; import sys; "
                "print(transcribe(sys.argv[1], Config()))")
        try:
            result=subprocess.run([sys.executable,'-c',script,str(audio)],
                                  capture_output=True,text=True,timeout=120,check=True)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            raise gr.Error('Speech recognition failed. Please try again or type your question.') from error
        text=result.stdout.strip()
        if not text:raise gr.Error('No speech was recognized. Try again or type your question.')
        return text
    dashboard_file=gr.File(visible=False, file_types=['.wav'], type='filepath')
    dashboard_api=gr.Button(visible=False)
    dashboard_api.click(dashboard_transcribe,dashboard_file,transcript,
                        api_name='transcribe_for_dashboard',api_visibility='public',
                        concurrency_id='workspace-speech-input',concurrency_limit=1)
    recording.start_recording(lambda:status_html('Recording','Speak your question, then stop recording. Transcription starts automatically.','working'),outputs=status,queue=False,api_visibility='private',
        js="() => {document.querySelector('#vb-playback audio')?.pause(); return []}")
    recording.stop_recording(lambda:status_html('Processing speech','Waiting for the completed audio file. If no transcript appears, check microphone permission or select Transcribe & review.','working'),outputs=status,queue=False,api_visibility='private')
    recording.clear(lambda:('',None,status_html('Ready to record','Microphone recording starts only when you choose Record.')),[ ],[transcript,cache,status],queue=False,api_visibility='private',cancels=jobs)
    def clear_voice(request=None):
        session_id=getattr(request,'session_hash',None)
        for key in list(voice.recent):
            if key[0]==session_id:voice.recent.pop(key,None)
        return None,'',None,''
    clear_voice.__annotations__={'request':gr.Request}
    clear_button.click(clear_voice,outputs=[recording,transcript,cache,draft],queue=False,api_visibility='private',cancels=jobs)
