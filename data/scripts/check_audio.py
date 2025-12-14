import os, traceback
p = os.path.join('data','fairseq','audio','rock_rock.00046.wav')
print('path:', os.path.abspath(p))
print('exists:', os.path.exists(p))
if os.path.exists(p):
    st = os.stat(p)
    print('size:', st.st_size)
    try:
        import soundfile as sf
        info = sf.info(p)
        print('soundfile info:', info)
        data, sr = sf.read(p, frames=1024)
        print('read_ok len', len(data), 'sr', sr)
    except Exception as e:
        print('soundfile error:')
        traceback.print_exc()
else:
    print('file missing')
