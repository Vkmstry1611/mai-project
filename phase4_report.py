"""Actual Phase 4 development and final holdout metrics; no policy selection."""
import argparse
from collections import Counter
import json
import os
from phase4_policy import run_path, verify_previous, verify_policy, P3
from phase3_common import write_new, sha, ROOT


def report(path):
    root=run_path(path);policy=verify_policy(root);verify_previous()
    if (root/'PHASE4_FINAL_REPORT.md').exists():raise FileExistsError('Completed report preserved')
    dev=json.loads((root/'development_summary.json').read_text())
    hold=json.loads((root/'holdout/summary.json').read_text()) if (root/'holdout/summary.json').exists() else None
    if hold is None:raise RuntimeError('Holdout is incomplete; do not label a partial run final')
    os.environ['MPLCONFIGDIR']=str(root/'runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plots=root/'plots';plots.mkdir(exist_ok=True)
    comparisons=dev['comparisons']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,key,label in [(axes[0],'vqa_percent','Development VQA agreement (%)'),(axes[1],'answer_coverage_percent','Development answer coverage (%)')]:
        vals=[r[key] for r in comparisons];bars=ax.bar(['Phase 2','A gate','Both gates','Balanced'],vals,color=['#527da3','#72908d','#827698','#4f8587'])
        ax.set_ylabel(label);ax.set_ylim(0,100)
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.2f}',ha='center')
    fig.tight_layout();fig.savefig(plots/'development_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    values=[hold['baseline_blip_v2_vqa_percent'],hold['frozen_phase2_vqa_percent'],hold['selected']['vqa_percent']]
    bars=ax.bar(['BLIP v2 control','Frozen Phase 2','Frozen Phase 4'],values,color=['#527da3','#72908d','#827698'])
    ax.set_ylabel('Official holdout VQA agreement (%)');ax.set_ylim(0,100)
    ax.set_title('Same single capture, original 3,319 held-out examples')
    for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.2f}',ha='center')
    fig.tight_layout();fig.savefig(plots/'holdout_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    sweep=dev['sweep'];ax.scatter([c['recall_percent'] for c in sweep],[c['vqa_percent'] for c in sweep],s=12,alpha=.25,label='Fixed grid')
    frontier=dev['pareto'];ax.scatter([c['recall_percent'] for c in frontier],[c['vqa_percent'] for c in frontier],s=18,label='Four-metric Pareto frontier')
    c=dev['selected'];ax.scatter(c['recall_percent'],c['vqa_percent'],s=100,marker='*',color='black',label='Balanced selection')
    ax.set_xlabel('Development answerability recall (%)');ax.set_ylabel('Development VQA agreement (%)');ax.legend();fig.tight_layout();fig.savefig(plots/'pareto.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    for ax,c,title in [(axes[0],dev['selected'],'Development (1,000)'),(axes[1],hold['selected'],'Holdout (3,319)')]:
        matrix=np.array(c['confusion_matrix_true_rows_predicted_columns_0_1']);ax.imshow(matrix,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
        ax.set_xticks([0,1],['Abstain','Answer']);ax.set_yticks([0,1],['Unanswerable','Answerable']);ax.set_xlabel('System decision');ax.set_ylabel('True label');ax.set_title(title)
    fig.tight_layout();fig.savefig(plots/'confusion_matrices.png',dpi=180);plt.close(fig)
    records=json.loads((root/'holdout/scored_records.json').read_text());devrecords=json.loads((root/'development_records.json').read_text())
    def categories(rows):
        result=Counter()
        for r in rows:
            abstain=r['final_answer'].strip().lower()=='unanswerable'
            if abstain:result['false_abstention_answerable' if r['answerable'] else 'correct_rejection_unanswerable']+=1
            else:result['retained_zero_credit' if r['final_vqa_score']<=1e-10 else 'retained_full_credit' if r['final_vqa_score']>=1-1e-10 else 'retained_partial_credit']+=1
            if r['new_abstention']:result['new_false_abstention' if r['answerable'] else 'new_rejected_unanswerable']+=1
        return dict(result)
    errors={'development':categories(devrecords),'holdout':categories(records),
        'holdout_beneficial_examples':[r for r in records if r['new_abstention'] and r['score_delta']>1e-10][:10],
        'holdout_harmful_examples':[r for r in records if r['new_abstention'] and r['score_delta']<-1e-10][:10],
        'holdout_incorrect_retained_examples':[r for r in records if r['final_answer']!='unanswerable' and r['final_vqa_score']<=1e-10][:10]}
    write_new(root/'error_analysis.json',errors)
    fields=['vqa_percent','precision_percent','recall_percent','f1_percent','answer_coverage_percent','abstention_percent']
    lines=['# VisionBridge Phase 4 final report','',
        'All values below are measured. Development selected a balanced policy; holdout was accessed only after policy and protocol freeze. No holdout threshold search, BLIP retraining or classifier fitting occurred.','',
        '## Why Phase 3 recall fell','',
        'The Phase 2 system already abstained on 288 answerable development questions. The heads cannot recover those answers. Joint gates added 94 answerable rejections and 20 unanswerable rejections. Recall fell from 57.58% to 43.74%. The reliability head has development average precision 0.3792 and treats partial-credit answers as negative in training. Its additional 60 rejections over the answerability gate included 56 answerable and four unanswerable questions, gaining only 0.38 VQA points. Twelve harmful abstentions came from the answerability cutoff and ten from the reliability cutoff. Among the joint policy\'s returned development answers, 101 received full VQA credit, 73 partial credit and 143 zero credit.','',
        'VQA agreement rewards the canonical unanswerable response when human references support it, even on some answerable-labelled examples. Return-versus-abstain precision is not answer correctness. Increased agreement with lower recall is therefore a trade-off, not a universal improvement.','',
        '## Development comparison (original 1,000 examples)','',
        '| Policy | A / R | VQA % | Precision % | Recall % | F1 % | Coverage % | Abstention % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in comparisons:lines.append('| '+c['name']+' | '+str(c['threshold_answerable'])+' / '+str(c['threshold_reliable'])+' | '+' | '.join(f'{c[k]:.2f}' for k in fields)+' |')
    lines+=['','## Balanced selection and freeze','',
        'Before replay, safeguards required nondecreasing VQA and answerability precision, at least 95% of Phase 2 recall, 90% of answer coverage and 90% of OCR-question coverage. These are explicit engineering choices, not validated clinical or accessibility standards. The fixed grid contains 240 gate combinations; ties favour recall, coverage and lower thresholds. Pareto dominance uses VQA, recall, precision and coverage. Equal-metric policies remain tied frontier entries.',
        f"Selected A={policy['threshold_answerable']:.2f}, R={policy['threshold_reliable']:.2f}, OCR=0.90. The A=0.00 cutoff effectively disables extra answerability rejection for valid probabilities; both heads still run. No evidence supports keeping the stricter gate as the balanced default.",
        f"Development: {dev['selected']['new_abstentions']} new abstentions, {dev['selected']['beneficial_changes']} beneficial, {dev['selected']['harmful_changes']} harmful, {dev['selected']['neutral_changes']} neutral. The 0.16-point gain is small and development-selected.",
        f"Frozen policy SHA256: `{sha(root/'selected-policy.json')}`. Heads: `{policy['models_sha256']}`. BLIP: `models/blip_vizwiz_v2`. Original manifest: `{policy['manifest_sha256']}`. Full source/checkpoint/scorer and decoding identities are in POLICY_FREEZE.json.",'',
        '## Final holdout (original 3,319 examples)','',
        'Only the frozen selected policy was evaluated. BLIP and Phase 2 controls are scored from the same single captures; no alternative gate settings were evaluated on holdout. The project training guide permits a final evaluation after model selection, and the current user explicitly authorized this stage. The 3,319 examples are the original project-held-out portion of VizWiz validation, not the official test partition.',
        '| Metric | Selected system |','|---|---:|']
    h=hold['selected']
    for k in fields:lines.append('| '+k.replace('_',' ')+' | '+f'{h[k]:.2f}'+' |')
    lines += [f"| Shared BLIP v2 control VQA % | {hold['baseline_blip_v2_vqa_percent']:.2f} |",f"| Shared frozen Phase 2 control VQA % | {hold['frozen_phase2_vqa_percent']:.2f} |",'',
        f"Historical Phase 1 holdout baseline: 59.48%. Current recapture difference: {hold['baseline_blip_v2_vqa_percent']-59.48:+.4f} points. A discrepancy, if present, must not be hidden or treated as an OCR improvement.",
        f"Selected minus shared Phase 2 holdout VQA: {h['vqa_percent']-hold['frozen_phase2_vqa_percent']:+.4f} points. New abstentions: {h['new_abstentions']}; beneficial {h['beneficial_changes']}, harmful {h['harmful_changes']}, neutral {h['neutral_changes']}. This is a measured paired difference, not a claim of statistical significance.",'',
        f"Agreement among returned replies: {h['answered_subset_vqa_percent']:.2f}%. Retained zero-credit replies: {h['answered_subset_zero_credit']}; partial-credit replies: {h['answered_subset_partial_credit']}. Overall VQA includes credit from abstentions and must not be interpreted as accuracy only among answers actually returned.",'',
        '## Subgroup agreement','', '| Split / subgroup | Examples | VQA % | Replies |','|---|---:|---:|---:|']
    for split,c in [('Development',dev['selected']),('Holdout',h)]:
        for name,g in c['subgroups'].items():lines.append(f"| {split} / {name} | {g['examples']} | {g['vqa_percent']:.2f} | {g['answered_examples']} |")
    lines+=['','### Balanced-policy development changes','']
    for r in [r for r in devrecords if r['new_abstention']][:10]:
        lines.append(f"- {r['image']}: {r['question']} — {r['phase2_final_answer']} → {r['final_answer']}; {r['phase2_vqa_score']:.2f} → {r['final_vqa_score']:.2f}; references {dict(Counter(a.get('answer') for a in r['answers']))}. {r['gate_reason']}")
    lines+=['','## Error analysis','',f"Development categories: `{errors['development']}`.",f"Holdout categories: `{errors['holdout']}`.",
        'False abstention means labelled answerable but declined; it need not be harmful by VQA references. Zero-credit answers are incorrect by agreement, not automatically proven visually unsupported. Full per-example references, evidence and decisions are preserved.','']
    for title,key in [('Beneficial holdout abstentions','holdout_beneficial_examples'),('Harmful holdout abstentions','holdout_harmful_examples'),('Incorrect retained answers','holdout_incorrect_retained_examples')]:
        lines+=['### '+title,'']
        for r in errors[key][:5]:lines.append(f"- {r['image']}: {r['question']} — {r['phase2_final_answer']} → {r['final_answer']}; {r['phase2_vqa_score']:.2f} → {r['final_vqa_score']:.2f}; refs {dict(Counter(a.get('answer') for a in r['answers']))}. {r['gate_reason']}")
        if not errors[key]:lines.append('No examples in this category.')
        lines.append('')
    seconds=sum(hold['timing_recorded_component_seconds'].values())
    lines+=['## Runtime, integration and verification','',
        f"Recorded inference/extraction/decision components: {seconds:.2f} seconds total, {seconds/3319:.3f} seconds/example; peak allocated GPU memory {hold['peak_vram_mb']:.1f} MiB. Dataset download and unrecorded feature/cache overhead are excluded. Execution wall time is recorded separately; resumed runs may span multiple executions.",
        f"Capture-loop wall time in the final execution: {hold['capture_wall_seconds_this_execution']:.2f} seconds ({hold['capture_wall_seconds_this_execution']/3600:.2f} hours), {hold['capture_wall_seconds_this_execution']/3319:.3f} seconds/example averaged across all 3,319 rows. This is not cold-start application latency; in a resumed execution this includes reused records and is not the total across all attempts.",
        'Use launch_phase4.py in .venv. It reuses app.py --phase4-run and the existing Gradio UI while routing model work to an application-only CUDA bootstrap in .venv-gpu. A device-only probe confirmed that CPU YOLO can hide CUDA before BLIP loads; initializing CUDA and caching explicit GPU resolution first prevents CPU fallback. The frozen benchmark already initializes CUDA before evidence, so its files and protocol were preserved. APPLICATION_FREEZE.json records the additional integration sources. An obsolete Phase 2 warning about missing learned answerability is removed from the application response. No environment packages were changed.',
        'BLIP v2 and unchanged OCR/object evidence feed the same gate features; learned scores and original candidate/evidence remain visible. Clear abstention asks for a clearer photo or another question. Optional speech uses a new request folder. The previous application mode remains available.',
        'Pre-holdout verification: 147 regression tests passed (four existing official-API regex deprecation warnings). Four additional application-bootstrap tests passed. Application feature/probability/decision parity passed on 25 cached development cases, including all OCR-source cases, with zero new neural calls. Gradio UI construction passed without starting a server. Two real device-only probes verified the CPU-YOLO CUDA side effect and bootstrap fix without loading another BLIP model. Post-evaluation regressions and the full image/capture/probability/scorer audit are recorded in execution/; these do not rerun neural inference. Preserved Phase 2 and Phase 3 hashes are rechecked. No checkpoint, manifest, old report or trained-source file was overwritten. No commit or push was performed.','',
        '## Implementation files','',
        '- `phase4_policy.py`: cached development search, safeguards, Pareto frontier and immutable policy/protocol verification.',
        '- `phase4_holdout.py`: exact verified images, resumable CUDA BLIP/CPU evidence capture and final single-policy scoring.',
        '- `phase4_pipeline.py`, `phase4_infer.py`, `app_phase4.py`: unchanged Phase 2/3 components under the selected policy, personal-photo CLI and UI bridge.',
        '- `app.py`: an explicit --phase4-run entry option; previous mode remains available.',
        '- `launch_phase4.py`, `phase4_gpu_infer.py`: application-only CUDA bootstrap and reuse of the existing controller, with a separate application source freeze.',
        '- `check_phase4.py`, `verify_phase4.py`, `verify_phase4_app_replay.py`, `tests/test_phase4_policy.py`: regressions and integration/artifact checks.',
        '- `run_phase4_completion.py`, `phase4_report.py`, `PHASE4_GUIDE.md`: automatic completion, actual results and execution instructions.','',
        '## Limitations and next steps','',
        '- The balanced gate has little practical effect; preserving coverage sacrifices most of the strict-policy VQA gain. Do not present it as a strong learned verifier.',
        '- Repeated development selection increases optimism. Holdout is now consumed for this final protocol; do not use its errors to retune this experiment.',
        '- Training predictions came from BLIP trained on those same examples. Classifier calibration cannot remove this optimism.',
        '- Gates cannot improve wrong answer content or recover existing abstentions. OCR support is restricted and ambiguity still retains BLIP.',
        '- Interactive accessibility/user testing, persistent GPU service latency and stronger independently trained evidence verification remain future work.',
        '- Any future method needs a fresh independent test protocol rather than iterating on these holdout outcomes.','',
        'Official source and scorer protocol: [VizWiz VQA](https://vizwiz.org/tasks-and-datasets/vqa/).',
        '[Development comparison](plots/development_comparison.png), [holdout comparison](plots/holdout_comparison.png), [Pareto frontier](plots/pareto.png), [confusion matrices](plots/confusion_matrices.png).']
    with (root/'PHASE4_FINAL_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    write_new(root/'FINAL_ARTIFACTS.json',{'report_sha256':sha(root/'PHASE4_FINAL_REPORT.md'),'holdout_summary_sha256':sha(root/'holdout/summary.json'),
        'holdout_records_sha256':sha(root/'holdout/scored_records.json'),'preserved_previous_integrity':True,'no_commit_or_push':True})
    print('PHASE4_FINAL_REPORT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();report(a.run_dir)
