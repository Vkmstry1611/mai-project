"""Capture exact permitted development fixture OCR; no BLIP/evaluation run."""
import json
from pathlib import Path
import time


def main():
    from PIL import Image
    from full_text_session import FullTextSession
    from reading_relevance import compose
    out=Path('reports/reading_sequences/run-20261020-v1')
    name='VizWiz_val_00002603.jpg'
    manifest=json.loads(Path('reports/phase2_local_dev_evaluation/drive_inputs/split_manifest.json').read_text())
    assert name in json.dumps(manifest['dev'])
    image=Image.open(Path('reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/data/images')/name).convert('RGB')
    s=FullTextSession('reports/phase4_policy/run-local-20261013-v1')
    try:
        times=[]
        for i in range(2):
            tick=time.perf_counter();response=s.reader.read(image,None);times.append(time.perf_counter()-tick)
            if i==0:
                with (out/'ORIGINAL_RAPID_RAW.json').open('x',encoding='utf-8') as f:json.dump(response,f,indent=2,ensure_ascii=False)
        before={q:{'route':s.classify(q).__dict__,'composition':compose(q,response['result'])}
                for q in ['Tell me the cooking instructions mentioned in the image?','Could you read the label?','Start with first passage.','1st passage.','Read every passage one by one.']}
        with (out/'BEFORE.json').open('x',encoding='utf-8') as f:json.dump({'questions':before,'raw_request_seconds':times,'startup':s.reader.startup,'fixture_split':'dev'},f,indent=2,ensure_ascii=False)
        print('Captured raw rows',len(response['result']['evidence']),'request seconds',times,flush=True)
        for q,result in before.items():print(q,result['route']['task'],repr(result['composition']['answer']),flush=True)
    finally:s.close()


if __name__=='__main__':main()
