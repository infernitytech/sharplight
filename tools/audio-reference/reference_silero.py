"""Reference tensors for the audio-model tests: run with the venv of the audio references (transformers, onnxruntime); writes into ~/models/audio-models/refs."""
import os
ROOT = os.path.expanduser("~/models/audio-models")
AUDIO = os.path.expanduser("~/models/audio")
REFS = ROOT + "/refs"
SILERO = os.environ["SILERO_VAD_PACKAGE"]  # the unpacked silero-vad wheel (silero_vad/): only its pure-Python segmentation is used
os.makedirs(REFS, exist_ok=True)
import sys, types, numpy as np, wave, onnxruntime as ort, json, importlib.util
src = open(SILERO + '/utils_vad.py').read()
i = src.index('def get_speech_timestamps_from_probs'); j = src.index('class VADIterator')
ns = {}
exec('from typing import List, Callable\nimport warnings\n' + src[i:j].replace('make_visualization', 'None'), ns)
seg = ns['get_speech_timestamps_from_probs']
sess = ort.InferenceSession(ROOT + '/silero-vad/silero_vad_16k_op15.onnx', providers=['CPUExecutionProvider'])
def probs(a):
    state = np.zeros((2,1,128), np.float32); ctx = np.zeros((1,64), np.float32); out=[]
    for s in range(0, len(a), 512):
        chunk = a[s:s+512]
        if len(chunk) < 512: chunk = np.pad(chunk, (0, 512-len(chunk)))
        x = np.concatenate([ctx, chunk[None]], 1).astype(np.float32)
        p, state = sess.run(None, {'input': x, 'state': state, 'sr': np.array(16000, np.int64)})
        ctx = x[:, -64:]
        out.append(float(p[0,0]))
    return out
for name in ('jfk', 'librispeech-1272-128104-0004'):
    w = wave.open(f'{AUDIO}/{name}.wav'); sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype('float32')/32768.0
    assert sr == 16000
    if a.ndim > 1: a = a.mean(1)
    p = probs(a)
    np.array(p, '<f4').tofile(f'{REFS}/silero_probs_{name}.f32')
    s = seg(p, sampling_rate=16000, audio_length_samples=len(a))
    json.dump(s, open(f'{REFS}/silero_segments_{name}.json', 'w'))
    print(name, len(a), len(p), s)
