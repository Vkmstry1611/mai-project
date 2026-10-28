"""CPU OCR miss/reader-reuse/cache-hit probe, not a Gradio latency benchmark."""
import json
from pathlib import Path
import time
import uuid


def main():
    from PIL import Image,ImageDraw,ImageFont
    from visionbridge.launch_persistent_demo import Session
    out=Path('reports/persistent_demo')/('ocr-profile-'+uuid.uuid4().hex);out.mkdir(parents=True,exist_ok=False)
    images=[]
    for index in range(2):
        image=Image.new('RGB',(600,250),'white')
        ImageDraw.Draw(image).text((70,70),'HELLO',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',90),fill='black')
        image.putpixel((0,0),tuple(bytes.fromhex(uuid.uuid4().hex[:6])))
        image.save(out/f'hello-{index}.png');images.append(image)
    session=Session('reports/phase4_policy/run-local-20261013-v1');records=[]
    try:
        for label,image in [('initial_reader_uncached_image',images[0]),('reused_reader_uncached_image',images[1]),('same_image_cache_hit',images[0])]:
            print('Starting OCR profile:',label,flush=True);tick=time.perf_counter()
            result=session.request(image,'What is written in this image?',True,True,'0')
            record=dict(label=label,callback_wall_seconds=time.perf_counter()-tick,answer=result[0],details=result[3],timings=result[-1])
            records.append(record)
            (out/(label+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
            if result[0]!='Recognized text fragments: HELLO.':raise RuntimeError('Real OCR HELLO regression; inspect saved result')
            print('Completed',label,round(record['callback_wall_seconds'],3),'seconds',flush=True)
        report=dict(records=records,gradio_benchmark=False,scope='Two distinct synthetic images then same-image repeat; CPU OCR and TTS; no BLIP inference or evaluation data.')
        (out/'RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Saved',out,flush=True)
    finally:session.close()


if __name__=='__main__':main()
