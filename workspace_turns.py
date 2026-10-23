"""Current question editing is separate from temporary conversation history."""


def question_after_submission(current,state):
    """None means leave the user's input alone (including newer typed text)."""
    state=state or {};submission=state.get('submission',{})
    image_id=state.get('conversation',{}).get('image_id')
    if submission.get('completed') and submission.get('image_id')==image_id and str(current or '').strip()==submission.get('question','').strip():
        return ''
    return None


def question_for_retake(current,state):
    return current if str(current or '').strip() else (state or {}).get('last_submitted_question','')
