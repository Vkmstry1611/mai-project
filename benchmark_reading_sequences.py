"""Sequential cold / fresh warm OCR / cached profiles on permitted fixtures."""
import argparse
import json
from pathlib import Path
import statistics
import time
from uuid import uuid4


def main(out):
    from PIL import Image
    from full_text_session import FullTextSession
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    root=Path('reports/phase4_policy/run-local-20261013-v1/application-requests')
    fixtures=[('cooking',Path('reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/data/images/VizWiz_val_00002603.jpg'),'Tell me the cooking instructions mentioned in the image?'),
              ('website',root/'071b6d6407ff4814ab6a6b46b08404e2/input.png','What website is in the image?'),
              ('blurry',root/'d57ffbb579ea495d86fb10c36cc87b93/input.png','What is written in the image?')]
    results=[]
    for name,path,q in fixtures:
        image=Image.open(path).convert('RGB');session=FullTextSession('reports/phase4_policy/run-local-20261013-v1')
        calls={'primary':0,'easy':0}
        for attr,key in [('reader','primary'),('easy','easy')]:
            worker=getattr(session,attr);old=worker.read
            def read(*args,_old=old,_key=key,**kw):calls[_key]+=1;return _old(*args,**kw)
            worker.read=read
        runs=[]
        def call(label,cache):
            tick=time.perf_counter();answer=session.full_request(image,q,evidence_cache=cache)
            entry={'label':label,'seconds':time.perf_counter()-tick,'answer':answer[0],'timings':answer[6],
                   'orientation':answer[3]['orientation'],'source':answer[3]['source']['engine'],'raw_calls':dict(calls)}
            runs.append(entry);print(name,label,round(entry['seconds'],3),dict(calls),flush=True)
            return entry
        try:
            cold=call('cold-fresh-workers',{})
            # Temporary namespaces force recognition again with the SAME loaded
            # models; do not confuse disk/raw cache hits with warm OCR inference.
            namespace='warm-'+uuid4().hex[:12]
            session.reader.cache_namespace=namespace;session.easy.cache_namespace=namespace
            cache={};warm=call('warm-models-fresh-recognition',cache)
            before=dict(calls)
            for n in range(3):
                cached=call('cached-followup-'+str(n+1),cache)
                assert cached['answer']==warm['answer'] and calls==before
            assert session.legacy.client.process is None
            values=[r['seconds'] for r in runs[2:]]
            results.append({'fixture':name,'runs':runs,'cached_mean':statistics.mean(values),'cached_median':statistics.median(values),
                'warm_cold_same_answer':warm['answer']==cold['answer'],'cached_no_extraction':calls==before})
            (out/'RESULTS.json').write_text(json.dumps({'results':results,'scope':'Experimental backend only, CPU OCR. Image opened before timer. TTS and Gradio excluded and measured in API tests. Runs are sequential. No old/new speedup claim; raw OCR behavior is unchanged.','holdout_inference':False},indent=2,ensure_ascii=False),encoding='utf-8')
        finally:session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='reports/reading_sequences/profile-'+uuid4().hex[:8]);main(p.parse_args().out)
