"""Permitted fixtures only: rendered known text, no VizWiz dataset access."""
import json
from pathlib import Path
import time
import re
import statistics
import uuid


def distance(a,b):
    previous=list(range(len(b)+1))
    for n,x in enumerate(a,1):
        current=[n]
        for m,y in enumerate(b,1):current.append(min(current[-1]+1,previous[m]+1,previous[m-1]+(x!=y)))
        previous=current
    return previous[-1]


def errors(reference,prediction):
    a=' '.join(reference.casefold().split());b=' '.join(prediction.casefold().split())
    return dict(cer=distance(a,b)/max(1,len(a)),wer=distance(a.split(),b.split())/max(1,len(a.split())))


def main():
    from PIL import Image,ImageDraw,ImageFont,ImageFilter
    from full_text_ocr import FullTextEngine
    from experimental_orientation import process_memory_mib
    out=Path('reports/full_text_reading')/('fixtures-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    text='VisionBridge reads printed text.\nThe label says FRAGILE. Handle with care.\nPlease keep this box dry.\nThank you for your help.'
    image=Image.new('RGB',(1100,390),'white')
    ImageDraw.Draw(image).multiline_text((40,45),text,font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',40),fill='black',spacing=18)
    fixtures={f'paragraph-{a}':(image.rotate(a,expand=True),text) for a in (0,90,180,270)}
    fixtures['small-text']=(image.resize((550,195)),text)
    fixtures['blank']=(Image.new('RGB',(600,400),'white'),None)
    fixtures['blurred']=(image.filter(ImageFilter.GaussianBlur(15)),None)
    fixtures['saved-photo']=(Image.open('reports/phase4_policy/run-local-20261013-v1/application-requests/9079468924344f5eb70fbc42e8ba4f22/input.png').convert('RGB'),None)
    engine=FullTextEngine();records={}
    for label,(photo,reference) in fixtures.items():
        photo.save(out/(label+'.png'))
        tick=time.perf_counter();result=engine.read(photo,out/'cache');elapsed=time.perf_counter()-tick
        records[label]=dict(seconds=elapsed,result=result,memory=process_memory_mib(),reference=reference,
            metrics=errors(reference,result['transcription']) if reference else None)
        with (out/(label+'.json')).open('x',encoding='utf-8') as f:json.dump(records[label],f,indent=2,ensure_ascii=False)
        print(label,round(elapsed,3),result['orientation']['selected_ccw_degrees'],repr(result['transcription']),records[label]['metrics'],flush=True)
    hit=time.perf_counter();cached=engine.read(fixtures['saved-photo'][0],out/'cache');cached_seconds=time.perf_counter()-hit
    repeats=[]
    for n in range(3):
        tick=time.perf_counter();result=engine.read(fixtures['saved-photo'][0],out/f'uncached-repeat-{n}');repeats.append(time.perf_counter()-tick)
    summary=dict(engine_loading_seconds=engine.loading_seconds,records=records,cache_hit_seconds=cached_seconds,
        warm_saved_photo_seconds=repeats,warm_mean_seconds=statistics.mean(repeats),warm_median_seconds=statistics.median(repeats),
        source=engine.source,scope='Known rendered text; photo lacks human-verified full transcript, so no photo CER/WER. CPU only.',
        gpu_vram_allocated_by_this_engine_mib=0)
    with (out/'RESULTS.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,ensure_ascii=False)
    print('FULL_TEXT_REPORT',out,flush=True)


if __name__=='__main__':main()
