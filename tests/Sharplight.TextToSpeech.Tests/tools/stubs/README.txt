Import stand-ins for the reference scripts of this folder (make_chatterbox_reference.py, make_chatterbox_clone_reference.py, make_f5tts_reference.py,
make_xtts_reference.py), which
put this folder first on sys.path: a minimal torchaudio (load, Resample, an HTK MelSpectrogram with torchaudio's buffer names, the mel helpers vocos
imports) for torch builds that have no torchaudio, empty wandb / datasets modules that the F5-TTS package imports for training only, and for
coqui-tts an importable torchcodec (it only checks that the package exists) and ko_speech_tools (Korean romanization, unused for English).
compliance.kaldi.fbank (make_chatterbox_clone_reference.py) delegates to torchaudio's own pure-torch compliance/kaldi.py, taken from a torchaudio
wheel and named by the environment variable SL_TORCHAUDIO_KALDI.
make_xtts_clone_reference.py needs the real torchaudio (MelSpectrogram with the Slaney norm, functional.resample): it puts the unpacked
torchaudio wheel named by SL_TORCHAUDIO_DIR ahead of these stand-ins.
