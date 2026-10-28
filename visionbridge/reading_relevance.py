"""Experimental transparent region selection. Never rewrites recognized words.

Raw extraction is immutable. Scores are layout heuristics, not probabilities.
RapidOCR's existing line acceptance (.75) and EasyOCR's .90 are unchanged.
"""
import math
import re
import statistics
import time

VERSION='question-aware-reading-v3-sequences'


def local_quality(rows,image_size):
    """Region heuristics complement global brightness/blur, never certify text."""
    heights=[];slopes=[]
    for r in rows:
        p=r.get('corrected_polygon') or r.get('polygon')
        if not p or len(p)!=4:continue
        heights.append(math.dist(p[0],p[3]))
        slopes.append(abs(math.degrees(math.atan2(p[1][1]-p[0][1],p[1][0]-p[0][0]))))
    hints=[]
    if heights and min(heights)<max(24,min(image_size)*.025):hints.append('Move closer to the small print and keep it in focus.')
    if slopes and statistics.median(slopes)>8:hints.append('Face the printed text toward the camera to reduce tilt and perspective.')
    return dict(heuristics=True,blocks_inference=False,minimum_detected_line_height=min(heights) if heights else None,
        median_text_skew_degrees=statistics.median(slopes) if slopes else None,warnings=hints,
        limitation='Detected regions only; missed text, curvature and recognition correctness cannot be measured by this heuristic.')


def intent(question):
    q=str(question).lower()
    from visionbridge.website_evidence import website_query
    if re.search(r'\b(?:read|written|transcribe)\b',q) and re.search(r'\b(?:product|nutrition) label\b',q):return 'all'
    if re.search(r'\b(?:how do i heat|heating instructions)\b',q):return 'heating'
    if re.search(r'\b(?:read|detect|extract|transcribe)\s+(?:every|all|everything)|\ball\s+(?:visible\s+)?(?:words|text)',q):return 'all'
    if website_query(q):return 'interface'
    if re.search(r'\b(?:what|which)\s+(?:web)?site\b|\b(?:url|address bar)\b|\bread\b.*\b(?:website|browser|toolbar|navigation)\b',q):return 'interface'
    from visionbridge.reading_sequences import instructions_query
    if instructions_query(q):return 'instructions'
    if re.search(r'\bsummari[sz]e|\bsummary\b|\b(?:explain|describe)\s+(?:(?:this|the|that)\s+)?(?:passage|article|text|document)\b|\bexplain in simple words\b|\bwhat\s+(?:is|was)\s+(?:this|the|that)\s+(?:passage|article|text|document)\s+about\b',q):return 'summary'
    if re.search(r'\b(?:story|passage|article)\b',q) and re.search(r'\b(?:what happened|who|where|when|why|how)\b',q):return 'text_question'
    return 'main'


def box(row):return row.get('corrected_box',row['box'])


def interface_kind(text):
    if re.search(r'https?://|www\.|\.(?:com|org|net)(?:\b|/)|\b\d{2,5}\s*[x×]\s*\d{2,5}\b|\.(?:gif|png|jpg)\b',text,re.I):return 'website_metadata'
    if re.search(r'^Safari\s+File.*View',text,re.I) or sum(bool(re.search(r'\b'+s+r'\b',text,re.I)) for s in ('Safari','File','Edit','View','History','Bookmarks','Window','Help'))>=3:return 'browser_toolbar'
    return None


def _geometric_regions(rows):
    """Connected aligned lines; remote columns and different font scales stay apart."""
    ordered=sorted(rows,key=lambda r:(box(r)[1],box(r)[0],r['id']));groups=[]
    for row in ordered:
        b=box(row);h=max(1,b[3]-b[1]);kind=interface_kind(row['text']);matches=[]
        for group in groups:
            last=group[-1];a=box(last);ah=max(1,a[3]-a[1]);gap=b[1]-a[3]
            overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))/max(1,min(a[2]-a[0],b[2]-b[0]))
            same_kind=kind==interface_kind(last['text'])
            if same_kind and min(h,ah)/max(h,ah)>=.55 and -min(h,ah)*.8<=gap<=2.5*max(h,ah) and (overlap>.45 or abs(a[0]-b[0])<1.5*max(h,ah)):
                matches.append((abs(gap),group))
        if matches:min(matches,key=lambda x:x[0])[1].append(row)
        else:groups.append([row])
    result=[]
    for n,group in enumerate(groups):
        heights=[max(1,box(r)[3]-box(r)[1]) for r in group]
        text='\n'.join(r['text'] for r in group);kind=interface_kind(text)
        accepted=[r for r in group if r.get('confidence',0)>=(.9 if r.get('source',{}).get('engine')=='EasyOCR' else .75)]
        coherence=sum(len(r['text'].split())>=3 for r in group)/len(group)
        alignment=1/(1+statistics.pstdev(box(r)[0] for r in group)/max(1,statistics.median(heights)))
        score=2*coherence+alignment+min(len(group),4)/4+len(accepted)/len(group)
        result.append(dict(id=f'region-{n}',kind=kind or ('main_passage' if coherence>=.5 and len(group)>=2 else 'text_region'),
            evidence_ids=[r['id'] for r in group],accepted_ids=[r['id'] for r in accepted],text=text,
            box=[min(box(r)[0] for r in group),min(box(r)[1] for r in group),max(box(r)[2] for r in group),max(box(r)[3] for r in group)],
            signals=dict(coherence=coherence,alignment=alignment,line_count=len(group),median_height=statistics.median(heights),accepted_fraction=len(accepted)/len(group)),score=score))
    # Screenshot chrome is supported jointly by multiple URL/menu detections,
    # a separate coherent lower passage and a marked font-size difference.
    anchors=[g for g in result if g['kind'] in ('browser_toolbar','website_metadata')]
    passages=[g for g in result if g['kind']=='main_passage' and g['signals']['line_count']>=3]
    if len(anchors)>=2 and passages:
        main=max(passages,key=lambda g:g['score'])
        for g in result:
            if g is not main and g['box'][3]<=main['box'][1]+.1*main['signals']['median_height'] and g['signals']['median_height']<.7*main['signals']['median_height']:
                g['kind']='browser_toolbar' if g['kind']=='text_region' or g['kind']=='main_passage' else g['kind']
    return result


def group_regions(rows):
    from visionbridge.reading_sequences import instruction_groups,accepted
    from modules.evidence_store import digest
    sequences=instruction_groups(rows);used={r['id'] for g in sequences for r in g['rows']}
    result=_geometric_regions([r for r in rows if r['id'] not in used])
    for sequence in sequences:
        group=sequence['rows'];ids=[r['id'] for r in group]
        heights=[max(1,box(r)[3]-box(r)[1]) for r in group]
        result.append(dict(id='sequence-'+digest(ids)[:16],kind=sequence['kind'],evidence_ids=ids,
            accepted_ids=[r['id'] for r in group if accepted(r)],text='\n'.join(r['text'] for r in group),
            box=[min(box(r)[0] for r in group),min(box(r)[1] for r in group),max(box(r)[2] for r in group),max(box(r)[3] for r in group)],
            signals=dict(line_count=len(group),median_height=statistics.median(heights),command_count=sequence['command_count'],
                observed_step_markers=sequence['observed_step_markers']),score=4.,grouping_reason=sequence['grouping_reason']))
    # OCR boxes jitter by a few pixels. Regions beginning on the same visual
    # row are ordered left-to-right, not by that detector jitter.
    bands=[]
    for g in sorted(result,key=lambda g:(g['box'][1],g['box'][0],g['id'])):
        h=g['signals']['median_height']
        if bands and abs(g['box'][1]-bands[-1]['top'])<=.5*min(h,bands[-1]['height']):
            bands[-1]['groups'].append(g)
        else:bands.append({'top':g['box'][1],'height':h,'groups':[g]})
    return [g for band in bands for g in sorted(band['groups'],key=lambda g:(g['box'][0],g['box'][1],g['id']))]


def compose(question,details,omit_ids=None,selected_region_ids=None):
    tick=time.perf_counter();rows=details.get('evidence',[]);regions=group_regions(rows);task=intent(question)
    if task=='heating':
        # Reuse geometrically grouped regions; never join a different column.
        groups=[g for g in regions if any(re.search(r'\bheating instructions\b',r['text'],re.I) for r in rows if r['id'] in g['accepted_ids'])]
        ids={i for g in groups for i in g['accepted_ids']} if len(groups)==1 else set()
        from visionbridge.reading_sequences import reading_order
        chosen=reading_order([r for r in rows if r['id'] in ids])
        # A nearby percentage or a microwave-time disclaimer is not a direction.
        directions=[r for r in chosen if re.search(r'\b(?:remove|pour|place|heat|stir|cover|microwave for|cook|simmer)\b',r['text'],re.I)]
        answer=('Recognized heating section (may be incomplete):\n'+'\n'.join(r['text'] for r in chosen)) if len(chosen)>1 else 'I cannot read sufficient heating directions reliably. Please show a close-up of the heating section; a heading alone is not enough.'
        if len(chosen)<=1 or not directions:
            answer='I cannot read sufficient heating directions reliably. Please show a close-up of the heating section; a heading alone is not enough.'
            chosen=[];ids=set()
        else:
            chosen=[r for r in chosen if re.search(r'[A-Za-z]{2}',r['text'])];ids={r['id'] for r in chosen}
            answer='Recognized heating section (may be incomplete):\n'+'\n'.join(r['text'] for r in chosen)
        return dict(details,question=question,answer=answer,final_answer=answer,reading_intent=task,regions=regions,
            selected_evidence=chosen,supporting_evidence_ids=sorted(ids),excluded_evidence=[r for r in rows if r['id'] not in ids],
            answer_source='ocr' if chosen else 'abstention',partial=True,decision='experimental_heating_section',composition_seconds=time.perf_counter()-tick)
    from visionbridge.text_hierarchy import build,request,select
    hierarchy=build(rows,regions);spec=request(question)
    details=dict(details,text_hierarchy=hierarchy)
    from visionbridge.ingredient_reading import intent as ingredient_intent,decide as ingredient_decide
    if ingredient_intent(question):
        answer,ids,trace=ingredient_decide(question,rows,regions)
        chosen=[r for r in rows if r['id'] in ids]
        return dict(details,question=question,answer=answer,final_answer=answer,
            selected_evidence=chosen,supporting_evidence_ids=sorted(ids),regions=regions,
            excluded_evidence=[r for r in rows if r['id'] not in ids],ingredient_evidence=trace,
            reading_intent='ingredients',answer_source='ocr' if chosen else 'abstention',
            decision='experimental_ingredient_retrieval',partial=True,composition_seconds=time.perf_counter()-tick)
    if spec:
        nodes,answer=select(spec,hierarchy,rows,omit_ids,selected_region_ids)
        ids={i for n in nodes for i in n['accepted_ids']}
        from visionbridge.reading_sequences import reading_order
        chosen=reading_order([r for r in rows if r['id'] in ids])
        return dict(details,question=question,answer=answer,final_answer=answer,
            selected_evidence=chosen,supporting_evidence_ids=sorted(ids),
            excluded_evidence=[r for r in rows if r['id'] not in ids],regions=regions,
            selected_hierarchy_ids=[n['id'] for n in nodes],hierarchy_request=spec,
            reading_intent='hierarchical',answer_source='ocr' if chosen else 'abstention',
            decision='experimental_hierarchical_reading',clarification_pending=not bool(chosen),
            partial=True,composition_seconds=time.perf_counter()-tick)
    from visionbridge.website_evidence import website_query,decide
    if task=='interface' and website_query(question):
        decision=decide(question,rows);ids=set(decision['ids']);chosen=[r for r in rows if r['id'] in ids]
        return dict(details,raw_transcription=details.get('transcription',''),question=question,answer=decision['answer'],final_answer=decision['answer'],
            selected_evidence=chosen,supporting_evidence_ids=sorted(ids),excluded_evidence=[r for r in rows if r['id'] not in ids],
            regions=[dict(g,selected=bool(set(g['evidence_ids'])&ids)) for g in regions],website_evidence=decision['candidates'],reading_intent=task,
            answer_source=decision['answer_source'],decision='experimental_website_retrieval',decision_reason=decision['reason'],
            abstention_reason=decision['reason'] if not ids else None,relevance_version=VERSION,partial=True,composition_seconds=time.perf_counter()-tick,
            relevance_selection_seconds=time.perf_counter()-tick-decision['composition_seconds'],answer_composition_only_seconds=decision['composition_seconds'])
    usable=[g for g in regions if g['accepted_ids']]
    browser={'browser_toolbar','website_metadata'}
    content=[g for g in usable if g['kind'] not in browser]
    candidates=[dict(id=g['id'],box=g['box'],accepted_ids=list(g['accepted_ids']),
        preview=' '.join(next(r['text'] for r in rows if r['id']==i) for i in g['accepted_ids'])[:120]) for g in content]
    ambiguous=False
    if selected_region_ids is not None:
        by_id={g['id']:g for g in content}
        selected=[by_id[i] for i in selected_region_ids if i in by_id]
    elif task=='interface':selected=[g for g in usable if g['kind'] in browser]
    elif task=='all':selected=usable
    else:
        sequences=[g for g in content if g['kind'] in ('instruction_sequence','ordered_list')]
        if task=='instructions' and re.search(r'\b(?:cooking|preparation|recipe)\b',question,re.I):
            sequences=[g for g in sequences if g['kind']=='instruction_sequence']
        passages=sorted([g for g in content if g['kind']=='main_passage'],key=lambda g:-g['score'])
        if sequences:
            selected=sequences if len(sequences)==1 else [];ambiguous=len(sequences)>1
        elif task=='instructions':selected=[]
        elif len(passages)>1 and abs(passages[0]['score']-passages[1]['score'])<.35:
            selected=[];ambiguous=True
        elif passages:selected=passages[:1]
        else:selected=content
    ids={i for g in selected for i in g['accepted_ids']}
    omitted=set(omit_ids or []);ids-=omitted
    from visionbridge.reading_sequences import reading_order
    chosen=reading_order([r for r in rows if r['id'] in ids])
    if task=='interface' and re.search(r'what (?:web)?site|which (?:web)?site',question,re.I):
        addresses=[r for r in chosen if re.search(r'https?://|www\.',r['text'],re.I)]
        chosen=addresses if len(addresses)==1 else []
        ids={r['id'] for r in chosen}
    partial=any(r['id'] not in ids and r['id'] not in omitted and not interface_kind(r['text']) for r in rows)
    selection_seconds=time.perf_counter()-tick;composition_started=time.perf_counter()
    if not chosen:
        answer=('I found multiple possible passages. '+ ' '.join(f'Passage {i+1}: {g["preview"][:65]}.' for i,g in enumerate(candidates))+
                ' Say first passage, second passage, or read all passages.' if ambiguous else
                'I cannot read sufficient reliable text for that request. Move closer and face the printed text toward the camera.')
    else:
        # Preserve exact recognized strings. No dictionary word splitting or punctuation repair.
        lines=[r['text'] for r in chosen]
        if task in ('summary','text_question'):
            # No generative language model is used. Specific factual questions need
            # independently verifiable relation reasoning, so do not fake an answer.
            if task=='text_question' and not re.search(r'what happened',question,re.I):
                answer='I cannot reliably answer that specific question from the recognized text. You can ask me to read the passage.';chosen=[];ids=set()
            else:
                # Do not stop a short wrapped passage after three physical lines.
                # Six lines cover more context without pretending to paraphrase it.
                limit=6 if task=='summary' else len(lines)
                answer=('The passage describes the following, as recognized in the image: ' if task=='summary' else 'The recognized passage says:\n')+(' '.join(lines[:limit]) if task=='summary' else '\n'.join(lines[:limit]))
                if task=='summary':answer+=' This reports what the passage says; it does not verify its claims. I cannot reliably explain details missing from the recognized text.'
                chosen=chosen[:limit];ids={r['id'] for r in chosen};partial=partial or limit<len(lines)
        elif task=='all':
            sections=[]
            for g in selected:
                group_lines=[r['text'] for r in chosen if r['id'] in g['accepted_ids']]
                if group_lines:sections.append(('Surrounding interface text' if g['kind'] in browser else 'Visible text region')+':\n'+'\n'.join(group_lines))
            answer='\n\n'.join(sections)
        else:answer=('Recognized cooking or preparation directions (partial):\n' if task=='instructions' else '')+'\n'.join(lines)
        if partial and chosen:answer+='\nSome smaller text or words could not be read clearly.'
    marked=[dict(g,selected=bool(set(g['accepted_ids'])&ids),selection_reason=('Question-relevant accepted evidence' if set(g['accepted_ids'])&ids else 'Surrounding interface, uncertain evidence, or a different region')) for g in regions]
    if not chosen and omitted:answer='There is no additional reliably recognized text in that reading scope. You can ask me to read every visible word for surrounding text.'
    return dict(details,raw_transcription=details.get('transcription',''),question=question,answer=answer,final_answer=answer,
        selected_evidence=chosen,supporting_evidence_ids=sorted(ids),excluded_evidence=[r for r in rows if r['id'] not in ids],
        regions=marked,reading_intent=task,relevance_version=VERSION,partial=partial or task=='instructions',
        passage_candidates=candidates,selected_region_ids=[g['id'] for g in selected],clarification_pending=ambiguous,
        answer_source='ocr' if chosen else 'abstention',composition_seconds=time.perf_counter()-tick,
        relevance_selection_seconds=selection_seconds,answer_composition_only_seconds=time.perf_counter()-composition_started,
        composition_note='Literal evidence only; no spelling completion. Summaries are explicitly extractive; unsupported specific questions abstain.')
