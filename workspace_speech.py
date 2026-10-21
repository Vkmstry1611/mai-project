"""Presentation-only speech: ordered chunks, one attempt per completed response."""
import atexit
from dataclasses import replace
from pathlib import Path
import re
import tempfile
import threading
import wave


def chunks(text, limit=1200):
    if not isinstance(text,str) or not text.strip():raise ValueError('No response to speak.')
    if limit<32:raise ValueError('Chunk limit is too small.')
    # Keep every non-whitespace character in order; never truncate a transcript.
    pieces=re.split(r'(?<=[.!?])\s+|\n+',text.strip())
    result=[];current=''
    for piece in pieces:
        while len(piece)>limit:
            cut=piece.rfind(' ',0,limit+1)
            if cut<=0:cut=limit
            head,piece=piece[:cut],piece[cut:].lstrip()
            if current:result.append(current);current=''
            result.append(head)
        if len(current)+len(piece)+1>limit:
            if current:result.append(current)
            current=piece
        else:current=(current+' '+piece).strip()
    if current:result.append(current)
    return result


def join_waves(paths,destination):
    params=None
    with wave.open(str(destination),'wb') as output:
        for path in paths:
            with wave.open(str(path),'rb') as source:
                shape=(source.getnchannels(),source.getsampwidth(),source.getframerate(),source.getcomptype())
                if params is None:
                    params=shape;output.setparams(source.getparams())
                elif params!=shape:raise ValueError('Speech chunks have incompatible audio formats.')
                while block:=source.readframes(65536):output.writeframesraw(block)
    return str(destination)


class ResponseSpeech:
    """Temporary output, bounded chunk size; each request creates its own files.

    No global text/audio cache, no cross-user memoization, no second synthesis
    after failure. Temporary directories are cleaned when this UI process exits.
    """
    def __init__(self,synthesizer=None):
        self.synthesizer=synthesizer;self.folders=[];self.lock=threading.Lock()
        atexit.register(self.close)

    def __call__(self,text):
        from config import Config
        from modules.text_to_speech import synthesize
        folder=tempfile.TemporaryDirectory(prefix='visionbridge-speech-')
        with self.lock:self.folders.append(folder)
        config=replace(Config(),output_dir=Path(folder.name))
        synth=self.synthesizer or synthesize
        paths=[synth(part,config) for part in chunks(text)]
        return paths[0] if len(paths)==1 else join_waves(paths,Path(folder.name)/'answer.wav')

    def close(self):
        with self.lock:
            for folder in self.folders:folder.cleanup()
            self.folders.clear()
