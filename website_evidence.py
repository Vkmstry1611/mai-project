"""Parse visible website evidence without resolving URLs or guessing characters."""
import re
import time
from urllib.parse import urlsplit

DOMAIN=re.compile(r'(?<![\w.-])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\s*\.\s*)+[a-z]{2,24}(?![\w-]|\.[a-z0-9])',re.I)
FILES={'gif','jpg','jpeg','png','svg','pdf','txt','html','css','js','json'}

def website_query(question):
    q=str(question).lower()
    return bool(re.search(r'\b(?:domain|url|web(?:site)? address|address bar)\b',q) or
                re.search(r'\b(?:what|which)\s+(?:(?:is|are)\s+)?(?:(?:the|this|that|a)\s+)?web\s*site\b|\b(?:read|identify)\s+(?:the\s+)?web\s*site\b',q) or
                re.fullmatch(r'\s*website(?:\s+(?:shown|visible|open))?\s*[?.!]*',q) or
                re.search(r'\b(?:what|which)\s+(?:is\s+)?(?:the\s+)?site\b.*\b(?:open|shown|visible|image)\b',q))

def candidates(rows):
    """Whitespace around observed punctuation is repairable; missing dots are not.

    Broken lines are joined only at an observed terminal dot with adjacent boxes.
    Every candidate carries all supporting raw IDs and its normalization record.
    """
    material=[(r['text'],[r]) for r in rows]
    ordered=sorted(rows,key=lambda r:((r.get('corrected_box') or r['box'])[1],(r.get('corrected_box') or r['box'])[0]))
    for a,b in zip(ordered,ordered[1:]):
        x=a.get('corrected_box',a['box']);y=b.get('corrected_box',b['box']);h=max(1,x[3]-x[1])
        if re.search(r'(?:https?://|www\.).*\.$',a['text'].strip(),re.I) and re.match(r'^[a-z]{2,24}(?:/|\b)',b['text'],re.I) and 0<=y[1]-x[3]<=1.5*h and abs(x[0]-y[0])<=2*h:
            material.append((a['text'].strip()+b['text'].strip(),[a,b]))
    found=[]
    browser_context=bool(re.search(r'Safari\s+File.*View|\bBookmarks\b.*\bWindow\b',' '.join(r['text'] for r in rows),re.I))
    extent=max(((r.get('corrected_box') or r['box'])[3] for r in rows),default=1)
    for text,source_rows in material:
        accepted=all(r.get('confidence',0)>=(.9 if r.get('source',{}).get('engine')=='EasyOCR' else .75) for r in source_rows)
        for match in DOMAIN.finditer(text):
            raw=match.group();domain=re.sub(r'\s*\.\s*','.',raw).lower()
            if domain.rsplit('.',1)[-1] in FILES:continue
            start=match.start();prefix=text[:start];tail=text[match.end():]
            scheme=re.search(r'https?://$',prefix,re.I)
            full=None
            if scheme:
                suffix=re.match(r'[^\s<>]*',tail).group().rstrip('.,;)')
                full=prefix[scheme.start():]+raw+suffix
                full=re.sub(r'\s*\.\s*','.',full)
                try:
                    if urlsplit(full).hostname!=domain:full=None
                except ValueError:full=None
            role='visible_domain'
            if full:
                role='embedded_image_source_url' if re.search(r'\.(?:gif|jpe?g|png|svg)(?:$|[?#])',urlsplit(full).path,re.I) else 'visible_url'
            elif prefix.endswith('@'):role='email_domain'
            elif re.search(r'https?://\S+\s+$',prefix,re.I):role='partial_domain_fragment'
            # A URL alone cannot establish address-bar ownership or active-page identity.
            found.append(dict(domain=domain,visible_url=full,raw_text=text,role=role,accepted=accepted,
                layout_role='possible_browser_address_row' if browser_context and (source_rows[0].get('corrected_box') or source_rows[0]['box'])[1]<extent*.4 else 'page_or_printed_content_unverified',
                evidence_ids=[r['id'] for r in source_rows],normalizations=['whitespace around observed dots'] if raw!=domain and any(c.isspace() for c in raw) else [],
                joined_lines=len(source_rows)>1,selection_signal='explicit_scheme' if full else 'observed_domain_only',
                limitation='OCR and layout do not verify the active website. No URL was fetched.'))
    return found

def decide(question,rows):
    started=time.perf_counter()
    all_candidates=candidates(rows);valid=[c for c in all_candidates if c['accepted']]
    explicit=[c for c in valid if c['visible_url']]
    direct=[c for c in valid if c['role'] not in ('email_domain','partial_domain_fragment')]
    pool=explicit or direct or valid;domains=sorted({c['domain'] for c in pool})
    ids=sorted({i for c in pool for i in c['evidence_ids']})
    selection_seconds=time.perf_counter()-started;composition_started=time.perf_counter()
    if not domains:
        titles=[r for r in rows if re.match(r'^(?:page|webpage|website) title\s*:',r['text'],re.I) and r.get('confidence',0)>=(.9 if r.get('source',{}).get('engine')=='EasyOCR' else .75)]
        if len(titles)==1:
            answer='The labeled page-title text reads: '+titles[0]['text'].split(':',1)[1].strip()+'. I cannot identify a website address from this title alone.'
            ids=[titles[0]['id']];reason='labeled_title_without_domain'
        else:
            answer='I could not find a reliable website address in the recognized text. Please show the address bar or printed web address more clearly.'
            reason='no_reliable_website_evidence'
    elif len(domains)>1:
        answer='Visible website evidence contains multiple domains: '+', '.join(domains)+'. I cannot determine which website is open.'
        reason='multiple_visible_domains'
    else:
        best=next(c for c in pool if c['domain']==domains[0]);want_url=bool(re.search(r'\b(?:url|address)\b',question,re.I))
        if want_url and best['visible_url']:
            answer='A visible URL reads: '+best['visible_url']+'. '
        elif best['role']=='email_domain':
            answer='A visible email address contains the domain '+domains[0]+'. '
        elif best['role']=='partial_domain_fragment':
            answer='I can only read this domain fragment: '+domains[0]+'. I cannot reconstruct the complete address. '
        else:
            answer=('A visible image-source URL contains the domain ' if best['role']=='embedded_image_source_url' else 'A visible web address contains the domain ')+domains[0]+'. '
        answer+='I cannot confirm that this identifies the website currently open in the browser.'
        reason='qualified_visible_website_evidence'
    return dict(answer=answer,ids=ids,candidates=all_candidates,reason=reason,answer_source='ocr' if ids else 'abstention',
                selection_seconds=selection_seconds,composition_seconds=time.perf_counter()-composition_started)
