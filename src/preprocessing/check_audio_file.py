import soundfile as sf
from pathlib import Path
p = Path('data/fairseq/audio/rock_rock.00085.wav')
print('exists', p.exists())
try:
    with sf.SoundFile(str(p)) as s:
        print('OK', s.frames, s.samplerate, s.channels, s.subtype)
except Exception as e:
    print('ERR', repr(e))
