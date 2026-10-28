"""Literal ingredient mentions from accepted OCR; no recipe knowledge or labels."""
import re
from visionbridge.reading_sequences import accepted,reading_order,COMMAND


def intent(question):
    q=str(question).lower().strip(' .?!')
    return bool(re.search(r'\bingredients?\b|\bwhat do i need to (?:prepare|cook|make)\b',q) or
                re.fullmatch(r'does (?:it|this|the (?:text|label|instructions)) (?:mention|contain) .{1,60}|is .{1,60} mentioned',q))


def decide(question,rows,regions):
    accepted_rows=reading_order([r for r in rows if accepted(r)])
    formal=[];mentions=[];ids=set()
    for g in regions:
        lines=[r for r in accepted_rows if r['id'] in g['accepted_ids']]
        if lines and re.match(r'^ingredients\s*:',lines[0]['text'],re.I):
            formal.extend(lines);ids.update(r['id'] for r in lines)
        elif lines and re.fullmatch(r'ingredients',lines[0]['text'].strip(),re.I):
            formal.extend(lines);ids.update(r['id'] for r in lines)
        if g['kind']!='instruction_sequence':continue
        for i,r in enumerate(lines):
            m=re.match(r'^(?:\d+[.)]\s*)?(?:brown|add|combine|stir in)\s+(.+)',r['text'],re.I)
            if not m:continue
            parts=[m[1]];sources=[r['id']]
            if re.search(r'\bstir in\b',r['text'],re.I):
                for following in lines[i+1:]:
                    if COMMAND.search(following['text']):break
                    parts.append(following['text']);sources.append(following['id'])
                    if following['text'].rstrip().endswith(('.', ';')):break
            phrase=' '.join(parts)
            phrase=re.split(r';|\b(?:until|into|in a|in the|for \d|to (?:a|the|boiling))\b',phrase,flags=re.I)[0]
            for value in re.split(r',|\band\b',phrase,flags=re.I):
                value=value.strip(' .;:')
                if not value or len(value.split())>7 or not re.search(r'[a-zA-Z]{2}',value):continue
                if re.search(r'\b(?:skillet|minutes?|occasionally|heat|boiling|cover)\b',value,re.I):continue
                if value.casefold() not in [x.casefold() for x in mentions]:mentions.append(value)
            ids.update(sources)
    q=str(question).strip(' .?!')
    yesno=re.fullmatch(r'does (?:it|this|the (?:text|label|instructions)) (?:mention|contain) (.+)|is (.+) mentioned',q,re.I)
    if yesno:
        term=re.sub(r'^the word\s+','',(yesno[1] or yesno[2]).strip(),flags=re.I);support=[r for r in accepted_rows if re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',r['text'],re.I)]
        # Presence requires accepted literal evidence. Absence is never certified
        # from incomplete OCR and never used for allergen/safety advice.
        if support:
            ids={r['id'] for r in support};answer='Yes. The recognized text says: '+ '; '.join('“'+r['text']+'”' for r in support)
        else:ids=set();answer='I did not find a reliable mention of '+term+' in the recognized text. OCR may be incomplete; this does not establish that it is absent.'
    elif formal and any(re.sub(r'^ingredients\s*:?','',r['text'],flags=re.I).strip() for r in formal):
        ids={r['id'] for r in formal};answer='Recognized ingredients section (may be incomplete):\n'+'\n'.join(r['text'] for r in formal)
    elif mentions:
        answer=('I cannot verify a formal ingredients list. ' if re.search(r'\b(?:list|listed)\b',q,re.I) else '')+'The preparation directions mention: '+', '.join(mentions)+'. These are preparation mentions, not the complete packaged-product ingredient list.'
    else:ids=set();answer='I cannot reliably extract ingredient names from this OCR evidence. Please provide a closer photo of the ingredients section.'
    return answer,ids,{'formal_section_detected':bool(formal),'preparation_mentions':mentions,'complete_ingredient_list_verified':False}
