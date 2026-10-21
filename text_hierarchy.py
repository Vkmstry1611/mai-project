"""Experimental source-only hierarchy and explicit scope retrieval.

Layout ordinals describe detected blocks, never recovered printed numbers.
No OCR thresholds, text, polygons or extraction caches are changed.
"""
import re
import statistics
from reading_sequences import accepted, bounds, reading_order, COMMAND, HEADING, marker
from modules.evidence_store import digest

ORD = dict(zip(('first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth'), range(1,11)))
ORD.update(dict(zip(('one','two','three','four','five','six','seven','eight','nine','ten'),range(1,11))))


def request(question):
    q=' '.join(str(question).lower().strip(' .?!').split())
    internal=re.fullmatch(r'(read|continue) detected step (step-[a-f0-9]{16})',q)
    if internal:return {'kind':'step','node_id':internal[2],'continue':internal[1]=='continue'}
    relative=relative_action(q)
    if relative:return {'kind':'step','relative':relative}
    if q in ('read all steps','read all cooking steps','read every step'):return {'kind':'step','all':True}
    if q in ('read the remaining steps','read remaining steps'):return {'kind':'step','rest':True}
    if q in ('read the remaining instructions','read remaining instructions'):
        return {'kind':'step','rest':True}
    if q in ('read remaining paragraphs','read remaining labels'):
        return {'kind':'paragraph' if 'paragraphs' in q else 'label','rest':True}
    m=re.search(r'\b(step|label|paragraph|heading)\b',q)
    if not m or not re.search(r'\b(read|repeat|continue|want|start)\b',q):return None
    kind=m[1]
    n=re.search(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2}(?:st|nd|rd|th)?)\b',q)
    if not n and kind!='heading':return None
    token=n[1] if n else 'first'
    number=ORD.get(token) or int(re.match(r'\d+',token)[0])
    return {'kind':kind,'number':number,'implicit_ordinal':n is None,'continue':bool(re.search(r'\bcontinue\b',q))}


def relative_action(question):
    q=' '.join(str(question).lower().strip(' .?!').split())
    if re.fullmatch(r'(?:(?:please|read|repeat|go to)\s+)*(?:the\s+)?previous (?:cooking )?step',q) or q=='go back one step':return 'previous'
    if re.fullmatch(r'(?:(?:please|read|go to)\s+)*(?:the\s+)?next (?:cooking )?step',q):return 'next'
    if q in ('repeat this step','repeat that step','read this step','repeat the current step'):return 'repeat'
    if q=='continue from here':return 'continue'
    return None


def navigation(question,context):
    action=relative_action(question)
    if not action:return None
    nodes=[n for n in context.get('text_hierarchy',{}).get('nodes',[]) if n['kind']=='step']
    cursor=context.get('step_cursor',{})
    current=next((n for n in nodes if n['id']==cursor.get('id')),None)
    if not current:return {'message':'Please select a supported cooking step first. I do not have a current step for this image.'}
    sequence=[n for n in nodes if n['sequence_id']==current['sequence_id']]
    index=next(i for i,n in enumerate(sequence) if n['id']==current['id'])
    target=index+({'previous':-1,'next':1}.get(action,0))
    if not 0<=target<len(sequence):
        return {'message':'You are already at the '+('first' if target<0 else 'last')+' detected instruction block. I will not wrap around or read the whole label.'}
    return {'question':('Continue' if action=='continue' else 'Read')+' detected step '+sequence[target]['id']}


def build(rows,regions):
    nodes=[]
    def node(kind,group,parent=None,**extra):
        ids=[r['id'] for r in reading_order(group)]
        value=dict(id=kind+'-'+digest(ids)[:16],kind=kind,parent_id=parent,evidence_ids=ids,
                   accepted_ids=[r['id'] for r in reading_order(group) if accepted(r)],
                   box=[min(bounds(r)[0] for r in group),min(bounds(r)[1] for r in group),
                        max(bounds(r)[2] for r in group),max(bounds(r)[3] for r in group)],**extra)
        nodes.append(value);return value
    if not rows:return {'version':1,'nodes':[]}
    image=node('image',rows)
    # OCR alone cannot certify multiple physical labels. One document container
    # is explicit; remote sections are not silently counted as separate labels.
    label=node('label',rows,image['id'],physical_label_count_verified=False)
    for region in regions:
        group=reading_order([r for r in rows if r['id'] in region['evidence_ids']])
        if not group:continue
        section=node('section',group,label['id'],region_id=region['id'])
        valid=[r for r in group if accepted(r)]
        if not valid:continue
        headings=[r for r in valid if HEADING.search(r['text']) or re.match(r'^you will need\b',r['text'],re.I)]
        for r in headings:node('heading',[r],section['id'])
        body=[r for r in valid if r not in headings]
        if not body:continue
        deltas=[bounds(b)[1]-bounds(a)[1] for a,b in zip(body,body[1:]) if bounds(b)[1]>bounds(a)[1]]
        pitch=statistics.median(deltas) if deltas else 0
        groups=[];reasons=[]
        for r in body:
            numbered=bool(marker(r['text']))
            gap=bounds(r)[1]-bounds(groups[-1][-1])[1] if groups else 0
            # Require both a conspicuous baseline gap and an instruction start.
            # In particular, adjacent commands ("Heat", "Cover") stay together.
            layout=bool(groups and pitch and gap>=1.5*pitch and COMMAND.search(r['text']))
            paragraph=bool(groups and pitch and gap>=1.8*pitch)
            if not groups or numbered or layout or paragraph:
                groups.append([]);reasons.append('observed_marker' if numbered else 'layout_gap_and_instruction' if layout else 'paragraph_gap' if paragraph else 'start')
            groups[-1].append(r)
        instruction=region['kind'] in ('instruction_sequence','ordered_list')
        reliable=(len(groups)>1 or any(marker(g[0]['text']) for g in groups)) and all(COMMAND.search(g[0]['text']) or marker(g[0]['text']) for g in groups)
        for g,reason in zip(groups,reasons):
            p=node('paragraph',g,section['id'],boundary_basis=reason)
            if instruction and reliable:
                observed=marker(g[0]['text'])
                node('step',g,p['id'],sequence_id=section['id'],boundary_basis=reason,
                     observed_marker=observed[0] if observed else None,
                     numbering_verified=bool(observed))
        for r in group:node('line',[r],section['id'])
    return {'version':1,'nodes':nodes,'limitation':'Physical label count and missing printed numbers are not verified by OCR.'}


def select(spec,hierarchy,rows,read_ids=None,region_ids=None):
    nodes=[n for n in hierarchy['nodes'] if n['kind']==spec['kind'] and n['accepted_ids']]
    if spec['kind']=='label' and nodes and any(not n.get('physical_label_count_verified') for n in nodes):
        return [],'I have one OCR document container, but cannot reliably distinguish separate physical labels. Do you mean the heading, a paragraph, or the first cooking step? Please specify, or crop the label you want read.'
    if region_ids and spec['kind'] in ('paragraph','heading'):
        parents={n['id']:n.get('parent_id') for n in hierarchy['nodes']}
        sections={n['id'] for n in hierarchy['nodes'] if n.get('region_id') in region_ids}
        def inside(n):
            p=n['id']
            while p:
                if p in sections:return True
                p=parents.get(p)
            return False
        nodes=[n for n in nodes if inside(n)]
    if spec['kind']=='step' and len({n['sequence_id'] for n in nodes})>1:
        return [],'I found multiple instruction sections. Please specify which section or crop the intended instructions.'
    if spec.get('implicit_ordinal') and len(nodes)>1:
        return [],f'I found {len(nodes)} possible headings. Please specify the first or second heading.'
    if spec.get('relative'):
        return [],'Please select a supported cooking step first so I can resolve that navigation request.'
    if spec.get('node_id'):
        positions=[i for i,n in enumerate(nodes) if n['id']==spec['node_id']]
        if len(positions)!=1:return [],'That step is no longer available for this image. Please select a supported step again.'
        chosen=nodes[positions[0]:] if spec.get('continue') else [nodes[positions[0]]]
    elif spec.get('all'):chosen=nodes
    elif spec.get('rest'):
        chosen=[n for n in nodes if not set(n['accepted_ids'])<=set(read_ids or [])]
    else:
        number=spec['number']
        numbered=[n for n in nodes if n.get('observed_marker')]
        if spec['kind']=='step' and numbered:
            matches=[i for i,n in enumerate(nodes) if n.get('observed_marker') and
                     (re.search(r'\d+',n['observed_marker']) and int(re.search(r'\d+',n['observed_marker'])[0])==number or
                      n['observed_marker'].strip() in '①②③④⑤⑥⑦⑧⑨⑩' and '①②③④⑤⑥⑦⑧⑨⑩'.index(n['observed_marker'].strip())+1==number)]
            if len(matches)!=1:return [],f'I cannot uniquely verify printed step {number}. Please identify its opening words or provide a closer crop.'
            number=matches[0]+1
        if not 1<=number<=len(nodes):
            return [],f'I cannot reliably select {spec["kind"]} {number}. I found {len(nodes)} supported {spec["kind"]} blocks. Please identify its opening words or upload a closer crop.'
        chosen=nodes[number-1:] if spec.get('continue') else nodes[number-1:number]
    if not chosen:return [],'There are no remaining reliably segmented instructions in this image.'
    ids={i for n in chosen for i in n['accepted_ids']}
    text='\n'.join(r['text'] for r in reading_order(rows) if r['id'] in ids)
    prefix=''
    if spec['kind']=='step' and any(not n['numbering_verified'] for n in chosen):
        prefix='Detected instruction block'+('s' if len(chosen)>1 else '')+' (printed step numbers were not recognized):\n'
    if spec['kind']=='label':prefix='One OCR document container; separate physical labels are not verified:\n'
    if spec['kind']=='paragraph':prefix='Detected paragraph block'+('s' if len(chosen)>1 else '')+' (layout-based boundaries):\n'
    return chosen,prefix+text+'\nThis is recognized text only; unclear or missing words are not reconstructed.'
