"""Writes the longer recordings of the Cohere Transcribe tests into an audio folder (16 kHz mono PCM16 WAV).

Usage (in a venv with numpy, soundfile and librosa; network access to the Hugging Face Hub):
    python make_cohere_transcribe_clips.py <audio dir>

- obama-45s.wav: the first 45 s of hf-internal-testing/dummy-audio-samples obama_first_45_secs.mp3 (a public-domain weekly address).
- mls-german-4.wav, mls-french-3.wav, mls-spanish-1.wav: the first 4 / 3 / 1 test utterances of facebook/multilingual_librispeech (CC BY 4.0)
  in German, French and Spanish, joined with 0.3 s of silence (60.3 s, 49.1 s, 16.3 s).
Their MLS transcripts go to mls-transcripts.json.
"""
import io
import json
import os
import sys
import urllib.request

import librosa
import numpy as np
import soundfile as sf

ROWS = "https://datasets-server.huggingface.co/rows?dataset=facebook/multilingual_librispeech&config={}&split=test&offset=0&length={}"
OBAMA = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"


def load(url: str) -> np.ndarray:
    data = urllib.request.urlopen(url).read()
    audio, rate = librosa.load(io.BytesIO(data), sr=16000, mono=True)
    return audio.astype(np.float32)


def main(out: str) -> None:
    transcripts = {}
    pause = np.zeros(int(0.3 * 16000), dtype=np.float32)
    for name, config, count in [("mls-german-4", "german", 4), ("mls-french-3", "french", 3), ("mls-spanish-1", "spanish", 1)]:
        rows = json.load(urllib.request.urlopen(ROWS.format(config, count)))["rows"][:count]
        clips = [load(row["row"]["audio"][0]["src"]) for row in rows]
        joined = np.concatenate([part for clip in clips for part in (clip, pause)][:-1])
        sf.write(os.path.join(out, name + ".wav"), joined, 16000, subtype="PCM_16")
        transcripts[name] = " ".join(row["row"]["transcript"] for row in rows)
    sf.write(os.path.join(out, "obama-45s.wav"), load(OBAMA), 16000, subtype="PCM_16")
    with open(os.path.join(out, "mls-transcripts.json"), "w") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
