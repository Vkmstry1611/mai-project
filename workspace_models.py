"""Frontend model choices; inference remains in the existing sessions."""
from threading import Lock

BLIP = 'BLIP v2'
QWEN = 'Qwen3-VL 4B'


class FrontendModels:
    choices = [BLIP, QWEN]

    def __init__(self, factory, default=BLIP, initial=None):
        self.factory = factory
        self.default = default
        self.sessions = {default: initial} if initial is not None else {}
        self.lock = Lock()

    def get(self, name=None):
        name = name or self.default
        if name not in self.choices:
            raise ValueError('Select BLIP v2 or Qwen3-VL 4B.')
        with self.lock:
            if name not in self.sessions:
                self.sessions[name] = self.factory(name)
            return self.sessions[name]

    def close(self):
        with self.lock:
            for session in self.sessions.values():
                session.close()
            self.sessions.clear()
