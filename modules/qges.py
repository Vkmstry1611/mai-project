"""Training-free lexical relevance; scores are similarities, never answer confidence."""
import re

def select(question, evidence, top_k=5):
    words = set(re.findall(r'\w+', question.lower())) - {'what', 'is', 'the', 'a', 'on', 'in', 'this'}
    ranked = []
    for item in evidence:
        terms = set(re.findall(r'\w+', item['text'].lower()))
        score = len(words & terms) / max(1, len(words | terms))
        ranked.append(dict(item, relevance=score))
    return sorted(ranked, key=lambda item: item['relevance'], reverse=True)[:top_k]
