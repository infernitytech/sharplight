"""Writes the reference tensors of the CosyVoice 3 tests (FunAudioLLM's CosyVoice code on torch, float32, CPU).

Usage (in a venv with torch, torchaudio, onnxruntime, transformers, hyperpyyaml, omegaconf, x-transformers, conformer, diffusers, openai-whisper,
librosa and scipy; the stubs folder next to this script supplies lightning, which Matcha-TTS imports for logging only):
    python make_cosyvoice3_reference.py <CosyVoice repository> <Fun-CosyVoice3-0.5B dir> <output dir> <reference wav (16 kHz)> "<reference text>"

Follows CosyVoice3.inference_zero_shot on the recording and writes raw little-endian files, each stage's input included so that every stage can be
compared on the reference's own input:
- front ends and the ONNX encoders: ref16.f32, ref24.f32 (the recording at 16 kHz and resampled to 24 kHz as load_wav does), tok_mel.f32
  [frames, 128] (whisper.log_mel_spectrogram), tokens.i32 (speech_tokenizer_v3.onnx), fbank.f32 [frames, 80] (mean-subtracted Kaldi filter bank),
  xvector.f32 [192] (campplus.onnx), prompt_feat.f32 [frames, 80] (the Matcha log-mel at 24 kHz), cut to twice the kept tokens;
- language model: text_tokens.i32, prompt_text_tokens.i32; a greedy chain teacher forced from the zero-shot rows (greedy.i32, greedy_logits.f32
  [steps, 6761]); a chain sampled with the release's RAS settings (seed 1986) for the next stages (speech.i32);
- flow: flow_noise.f32 [prompt + frames, 80] (the decoder's fixed noise, time-major) and flow_mel.f32 [frames, 80] for speech.i32;
- vocoder: hift_noise.f32 [samples, 9] (the causal source's fixed uniform table), hift_f0.f32 [frames], wav.f32 for flow_mel.f32;
- cosyvoice3.json: the lengths.
"""
import json
import os
import sys

import numpy as np
import torch


def save(out: str, name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    dtype = np.int32 if name.endswith(".i32") else np.float32
    np.ascontiguousarray(array.astype(dtype)).tofile(os.path.join(out, name))


def main(repo: str, model_dir: str, out: str, wav_path: str, reference_text: str) -> None:
    sys.path.insert(0, repo)
    sys.path.insert(1, os.path.join(repo, "third_party", "Matcha-TTS"))
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs"))
    # matcha.utils' __init__ imports Matcha's training helpers (lightning callbacks, hydra, rich); register the package without running it.
    import types  # noqa: E402
    matcha_utils = types.ModuleType("matcha.utils")
    matcha_utils.__path__ = [os.path.join(repo, "third_party", "Matcha-TTS", "matcha", "utils")]
    sys.modules["matcha.utils"] = matcha_utils
    import onnxruntime  # noqa: E402
    import scipy.io.wavfile  # noqa: E402
    import torchaudio  # noqa: E402
    import torchaudio.compliance.kaldi as kaldi  # noqa: E402
    import whisper  # noqa: E402
    from hyperpyyaml import load_hyperpyyaml  # noqa: E402

    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(4)
    info = {}

    # --- front ends (CosyVoiceFrontEnd._extract_speech_token / _extract_spk_embedding / _extract_speech_feat)
    rate, pcm = scipy.io.wavfile.read(wav_path)
    assert rate == 16000 and pcm.ndim == 1
    speech16 = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)
    speech24 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(speech16)
    save(out, "ref16.f32", speech16[0])
    save(out, "ref24.f32", speech24[0])
    feat = whisper.log_mel_spectrogram(speech16, n_mels=128)
    save(out, "tok_mel.f32", feat[0].T)
    option = onnxruntime.SessionOptions()
    option.intra_op_num_threads = 1
    tokenizer_session = onnxruntime.InferenceSession(os.path.join(model_dir, "speech_tokenizer_v3.onnx"), sess_options=option, providers=["CPUExecutionProvider"])
    tokens = tokenizer_session.run(None, {tokenizer_session.get_inputs()[0].name: feat.numpy(),
                                          tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten()
    fbank = kaldi.fbank(speech16, num_mel_bins=80, dither=0, sample_frequency=16000)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    campplus = onnxruntime.InferenceSession(os.path.join(model_dir, "campplus.onnx"), sess_options=option, providers=["CPUExecutionProvider"])
    xvector = campplus.run(None, {campplus.get_inputs()[0].name: fbank.unsqueeze(0).numpy()})[0].flatten()
    save(out, "fbank.f32", fbank)
    save(out, "xvector.f32", xvector)
    with open(os.path.join(model_dir, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")})
    prompt_feat = configs["feat_extractor"](speech24).squeeze(0).T
    info["tok_frames"] = int(feat.shape[2])
    info["all_tokens"] = int(len(tokens))
    info["fbank_frames"] = int(fbank.shape[0])
    info["feat_frames"] = int(prompt_feat.shape[0])
    token_len = min(prompt_feat.shape[0] // 2, len(tokens))
    tokens = tokens[:token_len]
    prompt_feat = prompt_feat[:2 * token_len]
    save(out, "tokens.i32", tokens)
    save(out, "prompt_feat.f32", prompt_feat)
    info["tokens"] = int(token_len)

    # --- language model (Qwen2LM.inference for CosyVoice3LM, zero-shot)
    tokenizer = configs["get_tokenizer"]()
    text = "The quick brown fox jumps over the lazy dog, and then it takes a long nap in the warm afternoon sun."
    prompt_text = "You are a helpful assistant.<|endofprompt|>" + reference_text
    text_tokens = tokenizer.encode(text, allowed_special=configs["allowed_special"])
    prompt_text_tokens = tokenizer.encode(prompt_text, allowed_special=configs["allowed_special"])
    save(out, "text_tokens.i32", text_tokens)
    save(out, "prompt_text_tokens.i32", prompt_text_tokens)
    info["text"] = text
    info["prompt_text"] = prompt_text
    llm, flow, hift = configs["llm"], configs["flow"], configs["hift"]
    llm.load_state_dict(torch.load(os.path.join(model_dir, "llm.pt"), map_location="cpu", weights_only=True), strict=True)
    flow.load_state_dict(torch.load(os.path.join(model_dir, "flow.pt"), map_location="cpu", weights_only=True), strict=True)
    hift_state = {k.replace("generator.", ""): v for k, v in torch.load(os.path.join(model_dir, "hift.pt"), map_location="cpu", weights_only=True).items()}
    hift.load_state_dict(hift_state, strict=True)
    llm.eval()
    flow.eval()
    hift.eval()

    with torch.inference_mode():
        all_text = torch.tensor([prompt_text_tokens + text_tokens], dtype=torch.int32)
        text_emb = llm.llm.model.model.embed_tokens(all_text)
        sos = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
        task = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
        prompt_speech = llm.speech_embedding(torch.tensor([tokens.tolist()], dtype=torch.int32))
        lm_input = torch.concat([sos, text_emb, task, prompt_speech], dim=1)
        min_len = int(len(text_tokens) * 2)
        greedy, logits_all = [], []
        cache = None
        step_input = lm_input
        for i in range(300):
            masks = torch.tril(torch.ones((1, step_input.shape[1], step_input.shape[1]), dtype=torch.bool))
            y_pred, cache = llm.llm.forward_one_step(step_input, masks=masks, cache=cache)
            logits = llm.llm_decoder(y_pred[:, -1])[0]
            logits_all.append(logits.clone())
            if i < min_len:
                logits[llm.speech_token_size] = -float("inf")
            top = int(logits.argmax())
            if top in llm.stop_token_ids:
                break
            greedy.append(top)
            step_input = llm.speech_embedding.weight[top].reshape(1, 1, -1)
        save(out, "greedy.i32", greedy)
        save(out, "greedy_logits.f32", torch.stack(logits_all))
        info["greedy_steps"] = len(logits_all)

        torch.manual_seed(1986)
        speech = list(llm.inference(text=torch.tensor([text_tokens], dtype=torch.int32), text_len=torch.tensor([len(text_tokens)], dtype=torch.int32),
                                    prompt_text=torch.tensor([prompt_text_tokens], dtype=torch.int32),
                                    prompt_text_len=torch.tensor([len(prompt_text_tokens)], dtype=torch.int32),
                                    prompt_speech_token=torch.tensor([tokens.tolist()], dtype=torch.int32),
                                    prompt_speech_token_len=torch.tensor([len(tokens)], dtype=torch.int32),
                                    embedding=torch.zeros(1, 0)))
        save(out, "speech.i32", speech)
        info["speech"] = len(speech)

        # --- flow (CausalMaskedDiffWithDiT.inference, non-streaming as CosyVoice3Model.tts with stream=False)
        frames = 2 * (len(tokens) + len(speech))
        save(out, "flow_noise.f32", flow.decoder.rand_noise[0, :, :frames].T)
        mel, _ = flow.inference(token=torch.tensor([speech], dtype=torch.int32), token_len=torch.tensor([len(speech)], dtype=torch.int32),
                                prompt_token=torch.tensor([tokens.tolist()], dtype=torch.int32), prompt_token_len=torch.tensor([len(tokens)], dtype=torch.int32),
                                prompt_feat=prompt_feat.unsqueeze(0), prompt_feat_len=torch.tensor([prompt_feat.shape[0]], dtype=torch.int32),
                                embedding=torch.from_numpy(xvector).unsqueeze(0), streaming=False, finalize=True)
        save(out, "flow_mel.f32", mel[0].T)
        info["mel_frames"] = int(mel.shape[2])

        # --- vocoder (CausalHiFTGenerator.inference, finalize=True)
        samples = mel.shape[2] * 480
        save(out, "hift_noise.f32", hift.m_source.l_sin_gen.sine_waves[0, :samples])
        f0 = hift.f0_predictor.to(torch.float64)(mel.to(torch.float64), finalize=True).to(mel)
        save(out, "hift_f0.f32", f0[0])
        wav, _ = hift.inference(speech_feat=mel, finalize=True)
        save(out, "wav.f32", wav[0])
        info["samples"] = int(wav.shape[1])
        scipy.io.wavfile.write(os.path.join(out, "reference.wav"), 24000, (wav[0].numpy() * 32767).astype(np.int16))

    with open(os.path.join(out, "cosyvoice3.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
