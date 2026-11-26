# Pending feed actions (fix/feed-official)

Proposed publish request (not committed as `publish-request.json`; replace that file with it to run):

```json
{
  "entries": [
    "gemma-3-1b-it:bf16-safetensors",
    "silero-vad-v5:onnx",
    "whisper-tiny:q5-1-whispercpp",
    "whisper-tiny:f32-safetensors",
    "gemma-4-31b-it:qat-w4a16-safetensors",
    "gemma-4-31b-it:bf16-safetensors"
  ],
  "channel": "stable",
  "promote": ["silero-vad-v5.onnx", "whisper-tiny.q5-1-whispercpp"],
  "validityDays": 30,
  "reason": "Official releases: Gemma 3 1B-it BF16, Silero VAD from Silero's GitHub release v6.2 (1.0.1: same package file as the mirror-built 1.0.0, official provenance), whisper-tiny F32 safetensors and the q5_1 package rebuilt to require Silero >= 1.0.1, Google's Gemma 4 31B-it BF16 and QAT W4A16 (partial). The two Gemma 4 31B packages run on the self-hosted runner."
}
```

Source sizes (variant + companions): gemma-3-1b-it bf16 2.0 GB, silero 1.3 MB, whisper-tiny q5_1 32 MB, whisper-tiny f32
155 MB (small pass); gemma-4-31b-it QAT W4A16 23.3 GB, BF16 62.6 GB (large pass, self-hosted runner; the BF16 build needs about
130 GB of disk). Nothing is above 200 GB.

Expected versions: `silero-vad-v5.onnx` 1.0.1 (manifest-only patch reusing the 1.0.0 file: the official ONNX is byte-identical and
builds the identical package), `whisper-tiny.q5-1-whispercpp` 1.0.1 or 1.1.0 (new dependency range), the others 1.0.0.

## Removals (run by the owner after the request above is published and promoted)

The mirror-built Silero VAD (1.0.0, built from `luvox-ai/silero-vad`), in both channels, with the feed signing key, then upload the
re-signed listings as the publish workflow does:

```sh
feedtool unlist silero-vad-v5.onnx 1.0.0 --feed out/ --key keys/feed.key.json --channel stable \
  --reason "Built from a community mirror; 1.0.1 is the same file from Silero's official GitHub release"
feedtool unlist silero-vad-v5.onnx 1.0.0 --feed out/ --key keys/feed.key.json --channel unstable \
  --reason "Built from a community mirror; 1.0.1 is the same file from Silero's official GitHub release"
```

(Unlisted versions stay resolvable, so installed whisper-tiny 1.0.0 packages keep working; their range [1.0.0,2.0.0) picks 1.0.1.)

`sd3.5-medium.fp16-diffusers-taesd3` 1.0.0 was uploaded from a community mirror but never listed in a channel, so there is nothing
to unlist. Its manifest is in the synced listings, so the rebuild from `stabilityai/stable-diffusion-3.5-medium` becomes 1.0.1
(an identical package file: a manifest-only patch whose version folder gets its own copy of the file) or 2.0.0 (other pinned
revision, different file). Only after that version is published, delete the orphaned 1.0.0 objects (deleting them earlier would
let the rebuild take version 1.0.0 again):

```sh
aws s3 rm --recursive "s3://$R2_BUCKET/models/packages/sd3.5-medium.fp16-diffusers-taesd3/1.0.0/" --endpoint-url "$R2_ENDPOINT"
```
