"""Presentation-only helpers; never alter the frozen response or evidence."""


def input_error(image, question):
    if image is None:
        return 'Upload a photo before asking a question.'
    if not isinstance(question, str) or not question.strip():
        return 'Type a question, or transcribe a recording first.'
    return None


def evidence_text(details):
    lines=['Selected evidence candidates; selection does not guarantee the final answer is correct.']
    support=set(details.get('supporting_evidence_ids',[]))
    for item in details.get('selected_evidence',[]):
        identifier=item.get('id','unknown')
        lines.append(f"{item.get('kind','evidence')}: {item.get('text','')} | ID: {identifier} | "
                     f"used as final support: {'yes' if identifier in support else 'no'} | box: {item.get('box','not recorded')}")
    if not details.get('selected_evidence'):
        lines.append('No selected evidence candidates.')
    for name,state in details.get('extraction_status',{}).items():
        lines.append(f"{name} extraction: {state.get('status','unknown')}")
    lines.append('Decision: '+str(details.get('phase4_gate_reason',details.get('decision_reason','not recorded'))))
    return '\n'.join(lines)


def with_evidence_text(response):
    # Preserve the original five values, including the same debug object.
    return (*response,evidence_text(response[3]))
