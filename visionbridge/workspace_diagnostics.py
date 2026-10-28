"""UI-boundary diagnostics: stack locations, never images/questions/evidence."""
import logging
import traceback
import uuid

LOG = logging.getLogger('visionbridge.workspace')


def report_failure(stage, error):
    reference = uuid.uuid4().hex[:12]
    # Do not log repr(error), local variables, request arguments or state: an
    # external worker/library exception can contain private recognized text.
    stack = ''.join(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n'
                    for frame in traceback.extract_tb(error.__traceback__))
    LOG.error('Workspace failure reference=%s stage=%s exception=%s\n%s',
              reference, stage, type(error).__name__, stack)
    return reference
