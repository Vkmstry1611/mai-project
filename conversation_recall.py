"""Experimental session recall, resolved before model routing; no inferred history."""
import re


def recall(question, context):
    q=' '.join(str(question).lower().replace('’',"'").strip(' .?!').split())
    turns=context.get('turns',[])
    history=context.get('question_history') or [t for t in turns if t.get('task')!='history']
    numbers={'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6}
    ordinal=re.search(r'\b(first|second|third|fourth|fifth|sixth|\d+(?:st|nd|rd|th)?)\s+(?:user\s+)?question\b',q)
    if ordinal and re.search(r'\b(my|ask|question)\b',q):
        token=ordinal[1];n=numbers.get(token) or int(re.sub(r'\D','',token))
        found=next((t for t in history if t.get('question_number')==n),None)
        # Older in-session state predates stable numbering.
        if found is None and history and all('question_number' not in t for t in history):
            found=history[n-1] if 0<n<=len(history) else None
        return ('Your '+token+' question was: '+found['question']) if found else 'That question is not available in this image session. I will not guess its wording.'
    if re.search(r'\bwhat did i ask\b|\b(?:my|the) (?:previous|last) question\b',q):
        return 'Your previous question was: '+history[-1]['question'] if history else 'There is no earlier question in this image session.'
    if re.search(r'\b(?:repeat|say|read)\b.*\b(?:your (?:previous|last) (?:answer|response)|that again)\b',q):
        return turns[-1]['answer'] if turns else 'There is no earlier answer in this image session.'
    if re.search(r'\bexplain your last answer\b',q):
        return ('My last answer was: '+turns[-1]['answer']+' I can repeat its supporting text; please specify which part needs explanation.') if turns else 'There is no earlier answer to explain.'
    if re.search(r'\brepeat (?:the )?previous (?:cooking )?step\b',q):
        # Repeat the last spoken step, not decrement the sequence cursor.
        t=context.get('last_step') or next((t for t in reversed(turns) if t.get('step_ids')),None)
        return t['answer'] if t else 'No previously spoken cooking step is available. Ask me to read a numbered step first.'
    return None
