"""Paired old/new speech from identical saved OCR evidence, no model inference."""
import json
from pathlib import Path
import time
import wave

def main():
    from visionbridge.workspace_speech import ResponseSpeech
    from visionbridge.reading_relevance import compose
    from visionbridge.investigate_reading_upgrade import OUT
    source=json.loads((OUT/'rapid-baseline.json').read_text(encoding='utf-8'));speaker=ResponseSpeech();records={}
    for label,value in source.items():
        result=value['result'];new=compose('What is written in this image?',result);pairs={}
        for kind,text in [('before',result['final_answer']),('after',new['final_answer'])]:
            tick=time.perf_counter();path=speaker(text);seconds=time.perf_counter()-tick
            with wave.open(path) as f:duration=f.getnframes()/f.getframerate()
            pairs[kind]={'tts_text':text,'generation_seconds':seconds,'audio_duration_seconds':duration,'characters':len(text)}
        records[label]=dict(pairs,selected_ids=new['supporting_evidence_ids'],excluded_ids=[r['id'] for r in new['excluded_evidence']],regions=new['regions'])
    with (OUT/'PAIRED_SPEECH.json').open('x',encoding='utf-8') as f:json.dump(records,f,indent=2,ensure_ascii=False)
    print({k:{n:{a:b for a,b in x.items() if a!='tts_text'} for n,x in v.items() if n in ('before','after')} for k,v in records.items()},flush=True)

if __name__=='__main__':main()
