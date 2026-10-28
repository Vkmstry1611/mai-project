"""Layout-supported instruction sequences; never supply undetected numbers/words."""
import re
import statistics

COMMAND=re.compile(r'^(?:\d{1,2}[.)]?\s+)?(?:brown|stir|heat|reduce|cover|uncover|sprinkle|bake|cook|preheat|microwave|boil|simmer|drain|remove|add|mix|pour|cool|serve|place|bring|whisk|fold|knead|chop|slice)\b',re.I)
HEADING=re.compile(r'\b(?:cooking|preparation|directions|instructions|recipe|method)\b',re.I)
MARKER=re.compile(r'^\s*(\d{1,2})[.)]?\s*$|^\s*([①②③④⑤⑥⑦⑧⑨⑩])\s*$')
INLINE_MARKER=re.compile(r'^\s*(\d{1,2})[.)]\s+(?=\S)')
SECTION_BREAK=re.compile(r'^(?:ingredients|nutrition|allergens|you will need|storage)\b',re.I)


def bounds(r):return r.get('corrected_box',r['box'])
def accepted(r):return r.get('confidence',0)>=(.90 if r.get('source',{}).get('engine')=='EasyOCR' else .75)
def marker(text):return MARKER.fullmatch(text) or INLINE_MARKER.match(text)


def instructions_query(question):
    return bool(re.search(r'\b(?:cooking (?:instructions|steps|directions)|(?:preparation|prepare) (?:instructions|steps|directions)|recipes?|cooking steps|numbered (?:instructions|steps|list)|ordered list|preparation directions)\b',str(question),re.I))


def reading_order(rows):
    """Close baseline fragments left-to-right; wrapped lines top-to-bottom."""
    lines=[]
    for r in sorted(rows,key=lambda r:(bounds(r)[1],bounds(r)[0],r['id'])):
        b=bounds(r);h=max(1,b[3]-b[1]);center=(b[1]+b[3])/2
        matches=[line for line in lines if abs(center-line['center'])<.30*min(h,line['height'])]
        if matches:min(matches,key=lambda l:abs(center-l['center']))['rows'].append(r)
        else:lines.append({'center':center,'height':h,'rows':[r]})
    return [r for line in lines for r in sorted(line['rows'],key=lambda r:(bounds(r)[0],r['id']))]


def instruction_groups(rows):
    """Conservative column lanes anchored by commands or actual ordered markers.

    The OCR may miss circled numbers entirely. Command sequences can still be
    selected, but no replacement step numbers are generated. Different columns
    stay separate even when they share ordinal numbers or font sizes.
    """
    rows=sorted(rows,key=lambda r:(bounds(r)[1],bounds(r)[0],r['id']))
    anchors=[r for r in rows if accepted(r) and COMMAND.search(r['text'])]
    lanes=[]
    for r in anchors:
        b=bounds(r);h=max(1,b[3]-b[1]);matches=[]
        for lane in lanes:
            a=bounds(lane[-1]);ah=max(1,a[3]-a[1])
            intervening=any(HEADING.search(t['text']) and a[3]<=bounds(t)[1]<b[1] and abs(bounds(t)[0]-b[0])<2*h for t in rows)
            if not intervening and abs(a[0]-b[0])<=1.8*max(h,ah) and -max(h,ah)<=b[1]-a[3]<=16*max(h,ah):matches.append(lane)
        if matches:min(matches,key=lambda lane:abs(bounds(lane[-1])[0]-b[0])).append(r)
        else:lanes.append([r])
    # Generic ordered lists may have no cooking verb. Require two actual markers
    # with distinct observed numbers and a spatially adjacent text body.
    marker_lanes=[]
    for r in rows:
        if not accepted(r) or not marker(r['text']):continue
        b=bounds(r);h=max(1,b[3]-b[1]);adjacent=[t for t in rows if t is not r and len(t['text'].split())>=2 and
            0<=bounds(t)[0]-b[2]<=4*h and abs((bounds(t)[1]+bounds(t)[3]-b[1]-b[3])/2)<h]
        if INLINE_MARKER.match(r['text']):adjacent.append(r)
        if not adjacent:continue
        lane=next((g for g in marker_lanes if abs(bounds(g[-1])[0]-b[0])<h and 0<=b[1]-bounds(g[-1])[3]<=20*h),None)
        if lane is None:marker_lanes.append([r])
        else:lane.append(r)
    for lane in marker_lanes:
        if len({r['text'].strip(' .)') for r in lane})>=2:lanes.append(lane)
    results=[];used=set()
    for lane in sorted(lanes,key=lambda g:(bounds(g[0])[1],bounds(g[0])[0])):
        if all(r['id'] in used for r in lane):continue
        h=statistics.median(max(1,bounds(r)[3]-bounds(r)[1]) for r in lane)
        left=statistics.median(bounds(r)[0] for r in lane);top=min(bounds(r)[1] for r in lane)
        heads=[r for r in rows if accepted(r) and HEADING.search(r['text']) and 0<=top-bounds(r)[3]<=3*h and abs(bounds(r)[0]-left)<=2*h]
        if len(lane)<2 and not heads:continue
        bottom=max(bounds(r)[3] for r in lane);selected=[];last_bottom=top
        for r in rows:
            b=bounds(r)
            if r['id'] in used or b[1]<top-.5*h or abs(b[0]-left)>2*h:continue
            if (HEADING.search(r['text']) or SECTION_BREAK.search(r['text'])) and b[1]>bottom:break
            rh=max(1,b[3]-b[1])
            if not marker(r['text']) and min(rh,h)/max(rh,h)<.5:continue
            if b[1]-last_bottom>3*h:continue
            selected.append(r);last_bottom=max(last_bottom,b[3])
        # Include a nearby heading only when the lane itself is grounded.
        selected=heads[-1:]+selected
        markers=[r for r in selected if accepted(r) and marker(r['text'])]
        commands=[r for r in selected if accepted(r) and COMMAND.search(r['text'])]
        if len(commands)<2 and not (heads and commands) and len(markers)<2:continue
        if not selected:continue
        ids={r['id'] for r in selected};used|=ids
        numbers=[marker(r['text'])[0].strip() for r in markers]
        results.append({'rows':reading_order(selected),'kind':'instruction_sequence' if commands else 'ordered_list',
                        'observed_step_markers':numbers,'command_count':len(commands),
                        'grouping_reason':'Aligned command/numbered sequence; only observed OCR wording and markers are retained.'})
    return results


def passage_reference(question,context):
    """Return an explicit existing scope, never facts derived from answer prose."""
    q=' '.join(str(question).lower().strip(' .?!').split())
    candidates=context.get('passage_candidates',[]);active=context.get('active_passage_ids',[])
    ordinals={'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,'seventh':7,'eighth':8,'ninth':9,'tenth':10}
    match=re.fullmatch(r'(?:please\s+)?(?:(?:start with|read|start at)\s+)?(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d{1,2}(?:st|nd|rd|th)?)(?:\s+(?:passage|region|one))?',q)
    if match:
        token=match[1];number=ordinals.get(token) or int(re.match(r'\d+',token)[0])
        if not candidates:return {'message':'There are no stored passage choices for this image yet. Ask me to read the label first.'}
        if not 1<=number<=len(candidates):return {'message':f'There are {len(candidates)} available passages. Choose a passage number from 1 to {len(candidates)}.'}
        return {'region_ids':[candidates[number-1]['id']],'question':'Read this passage.','restart':True}
    if q in ('read all passages','read every passage','read every passage one by one','read all passages one by one'):
        return {'region_ids':[g['id'] for g in candidates] or None,'question':'Read every visible word.','restart':True}
    if q in ('start from the beginning','start over','read from the beginning'):
        ids=active or ([candidates[0]['id']] if candidates else [])
        if not ids:return {'message':'Ask me to read the image first, then choose where to begin.'}
        return {'region_ids':ids,'question':'Read this passage.','restart':True}
    if q in ('continue','continue reading','read the rest','read rest') and active:
        return {'region_ids':active,'question':'Read this passage.','omit_ids':context.get('read_ids',[])}
    if q in ('continue','continue reading'):
        return {'message':'Please choose a passage first, or ask me to read all visible text.'}
    return None
