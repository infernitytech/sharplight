Import stand-ins for the reference scripts of this folder (make_chatterbox_reference.py, make_f5tts_reference.py), which put this folder
first on sys.path: a minimal torchaudio (load, Resample, an HTK MelSpectrogram with torchaudio's buffer names, the mel helpers vocos imports) for
torch builds that have no torchaudio, and empty wandb / datasets modules that the F5-TTS package imports for training only.
