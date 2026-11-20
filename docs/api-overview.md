# Public API overview

This is a short tour of the types an application uses. Every public type and member carries XML documentation (shipped as
`*.xml` next to the assemblies), which is the reference; this page shows how the pieces fit together.

## Assemblies

The core engine and everything an application calls to run a model live in `Sharplight.Core`, including the graph generation for
model architectures (adapters, namespace `Sharplight.Architectures`) and the `.slm` package format (namespace `Sharplight.Package`).
Model formats (importers) are separate assemblies, so the core never knows a file format: format names (GGUF, safetensors, ONNX,
diffusers, whisper.cpp, ...) appear only in `Sharplight.Import.*`. Most applications reference only the `Sharplight` package and
call `services.AddSharplight()`.

| Assembly | Role | Main entry points |
|---|---|---|
| `Sharplight.Core` | Graph IR, model definition and description, runtime and provider abstractions; the task layer on top of compiled models: text generation (the former `Sharplight.Llm`: sessions, sampling, continuous batching, chat, structured output, speculative decoding, embeddings, reranking), speech recognition and synthesis, image generation; tokenizers and Jinja chat templates; audio (WAV, resampling) and image (PNG, planar tensors) I/O | `LlmModel`, `ITextGenerator`, `TaskModel`, `TaskClientProvider`, `ISpeechRecognizer`, `ISpeechSynthesizer`, `IImageGenerator`, `ModelDescription`, `InferenceEngine`, `ModelService`, `EmbeddingModel`, `Reranker`, `WavFile`, `PngCodec` |
| `Sharplight.Import.Gguf`, `.HuggingFace`, `.Onnx`, `.WhisperCpp`, `.Diffusers`, `.PyTorch`, `.Kokoro`, `.Snac`, `.Chatterbox`, `.CosyVoice`, `.F5Tts`, `.Xtts`, `.SileroVad`, `.Ultralytics` | Format importers into the canonical model source | `GgufImporter`, `HuggingFaceImporter`, `OnnxImporter`, `WhisperCppImporter`, `DiffusersImporter`, `services.AddXxxImporter()` |
| `Sharplight.Video` | Video encoding and decoding through Sharplight's own LGPL FFmpeg build (shared libraries shipped with the package, bound with FFmpeg.AutoGen): MP4, WebM, MOV, Matroska, GIF, APNG; H.264, AV1, VP9, VP8, ProRes, FFV1 and the machine's hardware encoders; the core adds managed animated PNG and GIF as the fallback | `VideoEncoderCatalog`, `VideoDecoderCatalog`, `IVideoEncoder`, `IVideoDecoder`, `VideoClip`, `services.AddSharplightVideo()` |
| `Sharplight.Cpu`, `Sharplight.Vulkan`, `Sharplight.Metal`, `Sharplight.Cuda` (+ `Sharplight.Shaders`) | Execution providers (Metal: Apple GPUs on macOS, the same Slang kernels compiled to MSL; CUDA: NVIDIA GPUs through the driver API, the Slang kernels compiled to PTX with NVRTC from `Sharplight.Cuda.Runtime`, bf16 GEMMs optionally on cuBLASLt from `Sharplight.Cuda.Blas.<rid>`, see [cuda.md](cuda.md)) | `CpuExecutionProvider`, `VulkanProvider`, `MetalProvider`, `CudaProvider` |
| `Sharplight.Vulkan.MoltenVK` | MoltenVK for the Vulkan provider on macOS (native library only; inactive elsewhere) | none (referenced by `Sharplight.Vulkan`) |
| `Sharplight.Audio` | Native audio: decoders (WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus), encoders (WAV, Ogg Opus), playback and capture devices | `IAudioDecoder`, `IAudioEncoder`, `IAudioDeviceProvider`, `services.AddSharplightAudio()` |
| `Sharplight` | One registration entry point that wires up everything above | `services.AddSharplight()`, `SharplightDefaults.CreateServiceProvider()` |
| `Sharplight.Hosting` | Model hosting for servers and apps: a `ModelHost` from options or JSON (chat, embedding, reranker and task models on shared devices), placement, fit planning, status for health checks and cluster coordinators ([hosting.md](hosting.md)) | `IModelHostManager`, `ModelHostingOptions`, `HostedModelOptions`, `ModelHostingJson`, `HostStatus`, `FitPlanner`, `services.AddSharplightHosting()` |
| `Sharplight.AspNetCore` | OpenAI- and Anthropic-compatible HTTP endpoints for ASP.NET Core | `AddSharplightServer`, `UseSharplightServer`, `MapSharplightServer`, `AddSharplightEmbeddings`, `AddSharplightReranker`, `AddSharplightSpeechRecognizer` |
| `Sharplight.Hub` | Search and pull Hugging Face models into a local store ([model-hub.md](model-hub.md)) | `HubClient`, `ModelPuller`, `ModelStore` |
| `Sharplight.Extensions.AI` | `Microsoft.Extensions.AI` adapters | `SharplightChatClient`, `SharplightEmbeddingGenerator`, `SharplightSpeechToTextClient`, `SharplightTextToSpeechClient`, `SharplightImageGenerator` |

## Registration

Composition goes through Microsoft.Extensions.DependencyInjection; there is no static registry and no process-wide wiring. Every
library ships `IServiceCollection` extensions (C# 14 extension blocks, in the `Microsoft.Extensions.DependencyInjection` namespace, so
they need no extra `using`):

| package | registration | registers |
|---|---|---|
| `Sharplight.Core` | `AddSharplightTasks()` | `TaskClientProvider`, the built-in factories (text generation, speech recognition: prompted encoder-decoders such as Whisper, Moonshine and Canary, and frame-aligned CTC / transducer models such as Parakeet, image classification/embedding, object detection, depth estimation, latent-diffusion image generation, video diffusion generation), the `ModelLoader` |
| | `AddModelImporter<T>()`, `AddModelAdapter<T>()`, `AddCompositeModelAdapter<T>()`, `AddMultimodalArchitecture<T>()`, `AddAuxiliaryArchitecture(arch, role)` | one format / architecture (for importer and architecture packages) |
| | `AddTaskClientFactory<TFactory>()`, `AddTaskClientFactory<TClient>(kind, create, canCreate)` | one task implementation (for architecture packages) |
| | `AddExecutionProvider<T>()` | one execution provider (for provider packages) |
| | `AddTaskClient<TClient>(o => ...)`, `AddTextGenerator`, `AddSpeechRecognizer`, `AddSpeechSynthesizer`, `AddImageClassifier`, `AddImageEmbedder`, `AddObjectDetector`, `AddDepthEstimator`, `AddImageSegmenter`, `AddTextRecognizer`, `AddImageUpscaler`, `AddImageGenerator`, `AddVideoGenerator` (each also with `(o, services) => ...`) | a task client for one configured model |
| `Sharplight.Core` (`Sharplight.Architectures`) | `AddBuiltinArchitectures()` | the built-in adapters (incl. image encoders, image-text, DETR/RT-DETR, depth, latent diffusion), Qwen-VL prompt assembly, OuteTTS (composite adapter + speech-synthesis factory, WavTokenizer as its vocoder), the drafter architectures |
| `Sharplight.Import.Gguf` / `.HuggingFace` / `.Onnx` | `AddGgufImporter()`, `AddHuggingFaceImporter(o => ...)`, `AddOnnxImporter()` | the format (ONNX also its graph adapter) |
| `Sharplight.Import.WhisperCpp` | `AddWhisperCppImporter()` | whisper.cpp `ggml` model files |
| `Sharplight.Import.Kokoro` | `AddKokoroImporter()` | Kokoro-82M checkpoint folders (`config.json`, `.pth` with weight normalization, `voices/*.pt`, `lexicon/*.json`) |
| `Sharplight.Import.Snac` | `AddSnacImporter()` | SNAC audio codec checkpoint folders (the decoding half; auxiliary `vocoder` of Orpheus) |
| `Sharplight.Import.Diffusers` | `AddDiffusersImporter()` | diffusers folders and single-file Stable Diffusion checkpoints (ranks above the generic importers: `IModelImporter.Priority`) |
| `Sharplight.Import.StableAudio` | `AddStableAudioImporter()` | Stable Audio 3 checkpoint folders (`model_config.json`, `model.safetensors`, the T5Gemma folder) |
| `Sharplight.Import.Chatterbox`, `.CosyVoice`, `.F5Tts`, `.VoxCpm`, `.Xtts` | `AddChatterboxImporter()`, `AddCosyVoiceImporter()`, `AddF5TtsImporter()`, `AddVoxCpmImporter()`, `AddXttsImporter()` | the text-to-speech checkpoint layouts (F5-TTS also its Vocos vocoder, VoxCPM its AudioVAE `audiovae.pth`) |
| `Sharplight.Import.SileroVad` | `AddSileroVadImporter()` | Silero VAD checkpoints (voice activity detection) |
| `Sharplight.Import.Ultralytics` | `AddUltralyticsImporter()` | Ultralytics YOLO checkpoints and their graph adapter (AGPL-3.0 weights: see the model's license) |
| `Sharplight.Import.PyTorch` | `AddPyTorchImporter(o => ...)` | the restricted-unpickler settings (`TorchCheckpointOptions`) for pickle checkpoints (`pytorch_model*.bin` shards of the Hugging Face importer) |
| `Sharplight.Core` (`Sharplight.Package`) | `AddPackageImporter()` | the `.slm` importer and the `ModelConverter` |
| `Sharplight.Cpu`, `Sharplight.Vulkan`, `Sharplight.Metal`, `Sharplight.Cuda` | `AddCpuProvider()`, `AddVulkanProvider(o => ...)`, `AddMetalProvider(o => ...)`, `AddCudaProvider(o => ...)` | the provider (register GPUs first: discovery keeps the order; `AddSharplightProviders()` registers Metal, Vulkan, CPU, so Apple silicon prefers Metal over MoltenVK; the CUDA provider is opt-in and registered after Vulkan, which stays the default on NVIDIA GPUs) |
| `Sharplight.Audio` | `AddSharplightAudio(o => ...)` | native audio (own C library over miniaudio, libopus, libogg, opusfile, libvorbis): `IAudioDecoder` (WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus), `IAudioEncoder` (WAV, Ogg Opus), `IAudioDeviceProvider` (playback and capture streams); replaces the WAV-only `AddWavAudio()` services of the core, which stay the fallback when the native library cannot be loaded |
| `Sharplight.Core` | `AddVideoEncoders()` (also called by `AddSharplightTasks()`) | the `VideoEncoderCatalog` with the managed animated PNG and GIF encoders, and the (empty) `VideoDecoderCatalog` ([Video encoding](#video-encoding)) |
| `Sharplight.Video` | `AddSharplightVideo(o => ...)` | one `FfmpegVideoEncoder` per format (MP4, WebM, MOV, Matroska, GIF, APNG; preferred over the managed encoders) and the `FfmpegVideoDecoder` (`SharplightVideoOptions.EnableHardwareEncoders`), plus everything of `AddVideoEncoders()` |
| `Sharplight.Core` | `AddSharplightStorage()`, `AddSharplightStorage(layout)` | the `StorageLayout` (portable, system or custom location of configuration, data, models, caches and logs; [storage.md](storage.md)) |
| `Sharplight` | `AddSharplight()` | all of the above: `AddSharplightStorage()`, `AddSharplightModelFormats()`, `AddSharplightProviders()`, `AddSharplightTasks()`, `AddSharplightAudio()`, `AddSharplightVideo()` |
| `Sharplight.Extensions.AI` | `AddSharplightChatClient(o => ...)`, `AddSharplightEmbeddingGenerator(o => ...)`, `AddSharplightSpeechToTextClient()`, ... | Microsoft.Extensions.AI adapters |
| `Sharplight.Hosting` | `AddSharplightHosting(o => ...)`, `AddSharplightHosting("hosting.json")` | the `IModelHostManager` and its `ModelHost`, the `DeviceInventory`, the placement policy, the chat / embedding / reranker / task loader factories and the startup loader ([hosting.md](hosting.md)) |
| `Sharplight.AspNetCore` | `AddSharplightServer(host, o => ...)`, `AddSharplightServer(o => ...)` (the container's `ModelHost`), `AddSharplightEmbeddings`, `AddSharplightReranker` | the HTTP server (then `app.UseSharplightServer()`, `app.MapSharplightServer()`) |

## Video encoding

`VideoClip` (frames as `ImageInput`, frames per second, optional `Audio` track) is what video generation produces; `IVideoEncoder` writes it
in one file format and `VideoEncoderCatalog` (resolved from the container) picks the encoder by format id or file extension: the available
one with the highest `Priority`, so the bundled FFmpeg encoders win and the managed GIF / APNG encoders are the fallback when the FFmpeg
libraries are missing.

| format (`VideoFormats`) | extensions | codecs (`VideoCodecs`, default first) | audio |
|---|---|---|---|
| `mp4` | `.mp4`, `.m4v` | `h264` (OpenH264), `av1` (SVT-AV1), `vp9`; `h264-nvenc`, `hevc-nvenc`, `av1-nvenc`, `h264-videotoolbox`, `hevc-videotoolbox`, `h264-mf`, `hevc-mf` where they work | `aac`, `opus` |
| `webm` | `.webm` | `vp9`, `av1`, `vp8`; `av1-nvenc` | `opus` |
| `mov` | `.mov` | `prores` (proxy / lt / standard / hq as presets, 4:2:2 or 4:4:4 10-bit), `h264`, `ffv1`; VideoToolbox / NVENC | `aac`, `pcm` |
| `matroska` | `.mkv` | `h264`, `av1`, `vp9`, `ffv1` (lossless), `prores`; hardware encoders | `opus`, `aac`, `flac` |
| `gif` | `.gif` | FFmpeg `palettegen` / `paletteuse` (one optimized palette); managed `GifVideoEncoder` as the fallback | |
| `apng` | `.apng`, `.png` | FFmpeg APNG (lossless); managed `ApngVideoEncoder` as the fallback | |

```csharp
var services = new ServiceCollection().AddSharplight().BuildServiceProvider();   // or AddSharplightVideo() alone
var videos = services.GetRequiredService<VideoEncoderCatalog>();
var clip = new VideoClip(frames, framesPerSecond: 24) { Audio = soundTrack };      // frames: IReadOnlyList<ImageInput>, all the same size
await videos.SaveAsync(clip, "out.mp4", new VideoEncodeOptions { Codec = VideoCodecs.Av1, Preset = VideoPresets.Slow, Quality = 85 });
IReadOnlyList<VideoCodecInfo> choices = videos.FindForPath("out.mp4")!.Codecs;      // codecs, presets, pixel formats of this machine

var reader = services.GetRequiredService<VideoDecoderCatalog>();
VideoClip input = await reader.ReadClipAsync("in.mov", new VideoDecodeOptions { Start = TimeSpan.FromSeconds(2), FramesPerSecond = 16 });
```

`VideoEncodeOptions` is immutable: `Quality` (0 to 100, mapped to the codec's rate factor; dithering of GIF, effort of APNG),
`ConstantRateFactor` (the codec's own scale), `BitrateKilobitsPerSecond`, `Codec`, `Preset` (`VideoPresets` or the codec's own, see
`VideoCodecInfo.Presets`), `PixelFormat`, `FramesPerSecond` (retime), `KeyframeInterval`, `IncludeAudio` / `AudioCodec` /
`AudioBitrateKilobitsPerSecond`, `Loop` (GIF / APNG) and `MaxThreads`. A non-seekable output stream gets a fragmented MP4 / MOV.
`IVideoDecoder` (`FfmpegVideoDecoder`) probes files (`VideoInfo`), streams frames as RGB or I420 with timestamps (`VideoDecodeOptions`:
range, frame limit, frame-rate resampling, scaling) and decodes the audio track to an `AudioBuffer`. All work runs on the thread pool.

The FFmpeg libraries (avutil, swresample, swscale, avcodec, avformat, avfilter of FFmpeg 9.0) are built by `native/ffmpeg/build.sh` (LGPL
only, see `src/Sharplight.Video/THIRD-PARTY-NOTICES.md`), shipped as `runtimes/<rid>/native/` assets for win-x64, win-arm64, linux-x64,
linux-arm64 and osx-arm64, and loaded by full path (`BundledFfmpeg`; `SHARPLIGHT_FFMPEG_DIR` points at another build of the same major
versions). A system FFmpeg is never used.

All registrations are idempotent. Options are Microsoft.Extensions.Options with settable properties (`TaskLoadOptions`,
`LlmLoadOptions`, `EmbeddingLoadOptions`, `VulkanProviderOptions`, `HuggingFaceImportOptions`, `TorchCheckpointOptions`, `SharplightServerOptions`, ...), so they
also bind from configuration. What the container hands out:

- `ModelLoader` opens and builds models: `Importers` (`ModelImporterCatalog`: `Open`, `OpenSet`, `TryFind`) and `Architectures`
  (`ArchitectureCatalog`: `Build`, `TryFindAdapter`, `TryFindMultimodal`). Every load takes it: `TaskModel.Load(loader, options)`,
  `LlmModel.Load(loader, path, options)`, `EmbeddingModel.Load(loader, ...)`, `Reranker.Load(loader, ...)`, `SpanScorer.Load(loader, ...)`,
  `new LlmChatModelLoader(loader, ...)`.
- `ExecutionProviderCatalog` lists the providers (`Discover()` → `DiscoveredProvider(Provider, Availability)`, `TryFind(name)`,
  `TryFindForProfile(profile, out provider)` for offline precompilation); the container owns the providers.
- `ModelConverter` converts any loadable model to `.slm`; `ModelPuller` (Sharplight.Hub) takes one.

```csharp
var services = new ServiceCollection()
    .AddSharplight()                                                    // or pick: .AddGgufImporter().AddBuiltinArchitectures().AddCpuProvider()
    .Configure<VulkanProviderOptions>(o => o.EnableValidation = true)
    .AddTextGenerator((o, sp) =>
    {
        var gpu = sp.GetRequiredService<ExecutionProviderCatalog>().Discover().First(d => d.Availability.IsAvailable).Provider;
        o.ModelPath = "qwen3-4b-q4_k_m.gguf";
        o.Devices = [new DeviceSlot { Device = gpu.OpenDevice(gpu.GetDevices()[0]) }];
    });
await using var provider = services.BuildServiceProvider();
var generator = provider.GetRequiredService<ITextGenerator>();

// The lower-level model objects take the loader explicitly:
var loader = provider.GetRequiredService<ModelLoader>();
using var model = LlmModel.Load(loader, "qwen3-4b-q4_k_m.gguf", new LlmLoadOptions { Devices = [slot], ContextLength = 8192 });
```

Scripts, samples and tests without a host of their own use `SharplightDefaults.CreateServiceProvider()` (a disposable container with
`AddSharplight()`); a single-format tool can also build the catalogs by hand, e.g.
`new ArchitectureCatalog([new OnnxGraphAdapter()]).Build(new ModelSourceSet { Primary = OnnxImporter.Open(path) })`. The only
process-wide state left is the Vulkan instance (`VulkanProvider` remarks: devices opened by any provider object share it, and drivers
leak when instances are created repeatedly), so only the options of the provider that creates it take effect.

### Migrating from the static registries

| before | after |
|---|---|
| `GgufImporter.Register()`, `HuggingFaceImporter.Register()`, `OnnxImporter.Register()`, `ModelPackageImporter.Register()` | `services.AddGgufImporter()`, `.AddHuggingFaceImporter()`, `.AddOnnxImporter()`, `.AddPackageImporter()` (a new format: `services.AddModelImporter<MyImporter>()`) |
| `BuiltinAdapters.RegisterAll()`, `ModelAdapterRegistry.Register(a)`, `ModelAdapterRegistry.RegisterAuxiliaryOnly(arch, role)`, `MultimodalRegistry.Register(m)` | `services.AddBuiltinArchitectures()`, `.AddModelAdapter<T>()`, `.AddAuxiliaryArchitecture(arch, role)`, `.AddMultimodalArchitecture<T>()` |
| `ModelAdapterRegistry.RegisterComposite(a)` (an `IModelAdapter` with `CanLoad(ModelSourceSet)`) | `services.AddCompositeModelAdapter<T>()` with `T : ICompositeModelAdapter` (`CanLoad(sources)`, `Build(sources, options, architectures)`: component models are built with the catalog it gets) |
| `services.AddSharplightSpeechSynthesis()` (now in Sharplight.Core) | part of `services.AddBuiltinArchitectures()` |
| `CpuExecutionProvider.Register()`, `ProviderRegistry.Register(() => new VulkanProvider())`, `VulkanProvider.EnableValidation = true` | `services.AddCpuProvider()`, `.AddVulkanProvider(o => o.EnableValidation = true)` |
| `ModelImporterRegistry.Open/OpenSet(path)`, `ModelAdapterRegistry.Build(sources)` | `loader.Importers.Open(path)`, `loader.Open(path, auxiliary)`, `loader.Build(sources)` (or `loader.Build(path)`) |
| `ProviderRegistry.Discover()` (tuples), `ProviderRegistry.ForProfile(p)`, `devicePool.Open("vulkan", 0)` | `providers.Discover()` (`DiscoveredProvider`), `providers.TryFindForProfile(p, out var provider)`, `devicePool.Open(providers, "vulkan", 0)` |
| `DiffusersImporter.Register()`, `WhisperCppImporter.Register()` | `services.AddDiffusersImporter()`, `.AddWhisperCppImporter()` |
| `MultimodalRegistry.Find(arch)` | `model.Architectures.TryFindMultimodal(arch, out var m)`, or `visionEncoder.Multimodal` |
| `LlmModel.Load(path, o)`, `EmbeddingModel.Load(path, o)`, `Reranker.Load`, `SpanScorer.Load`, `TaskModel.Load(o)` | the same with the `ModelLoader` first |
| `ModelConverter.Convert(...)` (static), `new ModelPuller(hub, downloader, store)`, `PackagePrecompiler.Precompile(path, profile, o)` | `converter.Convert(...)` (service), `new ModelPuller(hub, downloader, store, converter)`, `PackagePrecompiler.Precompile(path, providers, profile, o)` |
| `Auxiliary = [("mmproj", path)]` | `Auxiliary = { [AuxiliaryRoles.Mmproj] = path }` (a dictionary, so it binds from configuration) |
| `services.AddSharplight(host, ...)`, `app.UseSharplight()`, `app.MapSharplight()` (AspNetCore) | `services.AddSharplightServer(host, ...)`, `app.UseSharplightServer()`, `app.MapSharplightServer()` |
| `services.AddSharplightChatClient(path, load, ...)` | `services.AddSharplightChatClient(o => { o.ModelPath = path; o.Devices = [...]; }, client => ...)` |

## Tasks: one model concept, one client pattern

Every model kind is the same thing to the engine: a `ModelDefinition` with graphs and weights plus a `ModelDescription` (kind and
components, see [model-format.md](model-format.md)), loaded through one load path (importer, then adapter, or the prebuilt definition
of an `.slm` package) into a `TaskModel`. What an application calls is a **task client**, and all kinds follow the same pattern:

| kind (`ModelKinds`) | client | whole result | streaming | request options | result |
|---|---|---|---|---|---|
| `text-generation` | `ITextGenerator` | `GenerateAsync(prompt)` | `GenerateStreamingAsync` → `GeneratedToken` | `GenerationOptions` | `TextGenerationResult` |
| `speech-recognition` | `ISpeechRecognizer` | `TranscribeAsync(AudioBuffer)` | `TranscribeStreamingAsync` → `TranscriptionSegment` | `SpeechRecognitionOptions` | `Transcription` |
| `speech-synthesis` | `ISpeechSynthesizer` | `SynthesizeAsync(text)` | `SynthesizeStreamingAsync` → `AudioBuffer` | `SpeechSynthesisOptions` | `SpeechSynthesisResult` |
| `image-generation` | `IImageGenerator` | `GenerateAsync(ImageGenerationRequest)` | `GenerateStreamingAsync` → `ImageGenerationUpdate` | `ImageGenerationOptions` | `ImageGenerationResult` |
| `video-generation` | `IVideoGenerator` | `GenerateAsync(VideoGenerationRequest)` | `GenerateStreamingAsync` → `VideoGenerationUpdate` | `VideoGenerationOptions` | `VideoGenerationResult` |
| `image-classification` | `IImageClassifier` | `ClassifyAsync(ImageInput)` | - (one image, one result) | `ImageClassificationOptions` | `ImageClassification` |
| `image-embedding` | `IImageEmbedder` | `EmbedImageAsync(ImageInput)`, `EmbedTextAsync`, `ClassifyZeroShotAsync` | - | `ImageEmbeddingOptions` | `float[]`, `ImageClassification` |
| `depth-estimation` | `IDepthEstimator` | `EstimateAsync(ImageInput)` | - | `DepthEstimationOptions` | `DepthMap` (per-pixel values, `ToGrayscale()`) |
| `object-detection` | `IObjectDetector` | `DetectAsync(ImageInput)` | - | `ObjectDetectionOptions` (score threshold, max detections, optional NMS) | `ObjectDetection` (`DetectedObject`: class, label, score, `BoundingBox` in pixels, `Mask` for instance segmentation models, `ReadingOrder` for document layout models: the objects then come back in reading order; `ObjectDetectionOptions.IncludeMasks`) |
| `image-segmentation` | `IImageSegmenter` | `EncodeImageAsync(ImageInput)` once per image, then `SegmentAsync(SegmentationImage, SegmentationPrompt)` per prompt (or `SegmentAsync(ImageInput, SegmentationPrompt)`) | - | `SegmentationOptions` (all candidate masks or the best one) | `ImageSegmentation` (`SegmentMask`: per-pixel mask at the image size, predicted IoU, area; object score) |
| `ocr` | `ITextRecognizer` | `RecognizeAsync(ImageInput)` | - | `TextRecognitionOptions` (max tokens, `Task`, `Input`) | `TextRecognition` (text, `TextRegion`s with optional boxes); also implemented by decoder-only vision-language recognizers (GLM-OCR: tasks `text`, `formula`, `table`). `DocumentReader(IObjectDetector layout, ITextRecognizer recognizer).ReadAsync(page)` runs layout, then OCR per region in reading order, and returns a `DocumentPage` of `DocumentRegion`s |
| `image-upscaling` | `IImageUpscaler` | `UpscaleAsync(ImageInput)` | `UpscaleOptions.Progress` reports finished tiles | `UpscaleOptions` (tile size, tile padding, progress) | `ImageInput` at `Scale` times the size |
| `audio-codec` | `IAudioCodec` | `EncodeAsync(AudioBuffer)` → codes, `DecodeAsync(CodecCodes)` → `AudioBuffer` | - | `AudioCodecOptions` (number of codebooks) | `CodecCodes` (codebook-major i32 codes) |
| `voice-activity` | `IVoiceActivityDetector` | `DetectAsync(AudioBuffer)` → segments, `ProbabilitiesAsync` | - | `VoiceActivityOptions` (thresholds, minimum speech and silence, padding, maximum segment) | `SpeechSegment` (start and end sample, `Start`, `End`) |

`SpeechSynthesisOptions` carries the voice: a built-in `Voice`, or a recording in `ReferenceAudio` (voice cloning) with its words in
`ReferenceText` for models that align them (F5-TTS, CosyVoice 3 zero-shot), and `Instructions`, a natural-language instruction on how to speak
for models that take one (CosyVoice 3: dialect, emotion, speed).
Multi-speaker models (VibeVoice) read a script of `Speaker N: text` lines and take one voice per speaker in `SpeakerVoices` (speaker label to `VoiceProfile`; `ReferenceAudio` and
`VoiceProfile` serve the first speaker); `DiffusionSteps` and `GuidanceScale` set the denoising steps per frame and the classifier-free guidance of models whose acoustic features come from a
diffusion head.

**Streaming synthesis.** `SynthesizeStreamingAsync` yields audio chunks while the model is still generating, so playback can start early. CosyVoice 3 runs
the flow with the chunk-causal attention of the reference's `stream=True` (a `cosyvoice_flow_estimator_chunked` graph next to the full-attention one; packages imported before
it existed stream one buffer per utterance): every 25 speech tokens it decodes the mel so far and the vocoder over it, and emits the samples no later frame can change (the
first chunk is about 0.85 s of audio). XTTS emits after 12 codes and then every 20 within a sentence (vocoder over the latents so far, four rows held back; the chunks add up to the
one-shot audio). Chatterbox emits chunks of speech tokens (the first after 15, then 25), each decoded by the flow and the vocoder with context tokens around it and cross-faded
with the previous chunk. The other models stream per sentence or word. Studio's Speak page plays the chunks back to back on one output stream and reports the time to the first audio.

**Saved voices.** A cloning model (`IVoiceCloner`, taken with `GetService(typeof(IVoiceCloner))`: Chatterbox, XTTS v2, CosyVoice 3, F5-TTS, VibeVoice) turns a recording into a `VoiceProfile`
once (`CreateVoiceAsync(name, recording, referenceText)`): the conditioning tensors the model derived (speaker embeddings, prompt tokens and mel, style latents; F5-TTS keeps the
recording itself with its words). `VoiceLibrary` stores them in the layout's `voices` data folder (`new VoiceLibrary(StorageLayout)`): one folder per voice with `voice.json` (name,
architecture family, reference text, where each tensor lies) and `tensors.bin`. A request passes `SpeechSynthesisOptions.VoiceProfile` instead of the recording and gets the same
audio, without running the encoders again; a voice of another model family is rejected. Studio's Speak page saves a voice from the reference recording (a file or the microphone),
lists the saved voices of the chosen model and deletes them.

**Text normalization.** `Sharplight.Speech.Normalization.TextNormalizers.For(language)` spells numbers, ordinals, dates, clock times, currency amounts, percentages and common
abbreviations as words for EN, DE, FR, ES, IT, PT, ZH and JA (`ITextNormalizer`). CosyVoice 3, Chatterbox, XTTS, F5-TTS and Orpheus apply it by default to the request's
`Language` (else the text's script); `SpeechSynthesisOptions.TextNormalizer` replaces it (for any model, Kokoro and OuteTTS included) and `NormalizeText = false` passes the
text as written.

Every client is an `ITaskClient`: `Metadata` (kind, model name, architecture), `GetService(Type)` for the objects behind it,
`IDisposable` (it owns its compiled graphs, not the model), async calls with a `CancellationToken`. The shape mirrors
Microsoft.Extensions.AI (`IChatClient`, `ISpeechToTextClient`, `ITextToSpeechClient`, `IImageGenerator`), and
`Sharplight.Extensions.AI` adapts the clients to those interfaces; Core itself does not depend on Microsoft.Extensions.AI, whose speech
and image abstractions are still experimental.

Implementations are `ITaskClientFactory` services (kind, client type, `CanCreate(model)`, `Create(model)`), resolved by the
`TaskClientProvider` from the model's kind; the last registration that accepts a model wins. Architecture packages register theirs
next to their adapters (`services.AddTaskClientFactory<MyRecognizerFactory>()`, or from delegates
`services.AddTaskClientFactory<ISpeechRecognizer>(ModelKinds.SpeechRecognition, model => ...)`); a new kind
(depth estimation, audio codecs, time series, ...) is a new kind string, client interface and factory, without changes to Core types.
Text generation is built in: `LlmModel` is the text-generation model and `LlmModel.CreateTextGenerator()` its client.

A client for a configured model is one registration; the model is loaded once on first use with `TaskLoadOptions` from
Microsoft.Extensions.Options (named per client, so keyed registrations can load different models) and disposed by the container
after the client:

```csharp
var services = new ServiceCollection()
    .AddSharplight()
    .AddTextGenerator(o => { o.ModelPath = "qwen3-4b-q4_k_m.gguf"; o.Devices = [slot]; o.ContextLength = 8192; })
    .AddSpeechRecognizer(o => { o.ModelPath = "whisper-small.slm"; o.Devices = [slot]; }, serviceKey: "whisper");
await using var provider = services.BuildServiceProvider();

var generator = provider.GetRequiredService<ITextGenerator>();
TextGenerationResult completion = await generator.GenerateAsync("The capital of France is", new GenerationOptions { MaxNewTokens = 32 });

var recognizer = provider.GetRequiredKeyedService<ISpeechRecognizer>("whisper");
Transcription text = await recognizer.TranscribeAsync(WavFile.Read("talk.wav"));
```

Without task registrations: `using var model = TaskModel.Load(loader, new TaskLoadOptions { ModelPath = path, Devices = [slot] })`, then
`taskClientProvider.Create<ISpeechRecognizer>(model)`. Task implementations use `TaskModel.CompileComponent(role)` (compiles a
component graph with the component's default bounds, consulting the package's precompiled artifacts), `CreateTokenizer(role)`,
`Metadata(role)` and `ReadAttachment(role, name)`.

Audio and image I/O for the task layer: `AudioBuffer` (interleaved float samples; `ToMono`, `Resample`, `ToModelInput`),
`AudioResampler` (band-limited windowed sinc), `WavFile` (read: integer PCM 8/16/24/32-bit and float 32/64-bit, plain or
`WAVE_FORMAT_EXTENSIBLE`; write: PCM16, PCM24, PCM32, float32), `PngCodec` (encode RGB; decode 8-bit gray, gray+alpha, RGB, RGBA and
palette PNGs) and `PlanarImage` (8-bit interleaved RGB to and from `[3, height, width]` float planes in any value range).

### Threading contract

Task clients are safe to call from a UI thread, and implementations (built-in or registered) keep this contract:

- **Every `...Async` method returns to its caller before any compute.** Compiling graphs, running them, tokenizing, decoding audio
  or images and encoding results run on a background thread; built-in clients use dedicated (long-running) threads, since a graph
  run blocks on device fences and would starve the thread pool. `Task.FromResult` over synchronous work is not allowed.
- **Streams are produced by a background loop** that queues its items for the consumer and never waits for it (a slow consumer, such
  as a UI that renders every preview, does not slow generation down), and the consumer's `MoveNextAsync` never runs a step itself.
- **Cancellation** is observed between steps (a denoising step, a decoded audio window, a generated token). A cancelled or abandoned
  enumeration (`break`, dispose) completes only after the step in progress, so once it has ended the client, and the model behind
  it, are idle and may be disposed or unloaded. Exceptions of the producer surface at the consumer after the items produced before.
- **No synchronization context is captured**: library code awaits with `ConfigureAwait(false)` throughout; results, progress
  callbacks (`IProgress<T>`, `Log`) and stream items arrive on background threads, and applications marshal only state updates to
  their UI thread (throttled: progress can be reported per kernel, per tensor or per token).
- **One request at a time per client**: built-in clients serialize requests (later ones wait for the earlier). Disposing a client while
  a request runs is not supported.
- `TaskModel.LoadAsync` loads on a background thread; the synchronous `TaskModel.Load` and `CreateTokenizer`/`CompileComponent` block
  and belong on a worker, never on a UI thread. Synchronous serving calls (`ModelService.StartChat`, which builds and tokenizes the
  prompt) likewise.

Sharplight Studio follows the same rule on its side: every page runs engine work on workers (the run queue, `Task.Run` around service
calls), posts only state changes to the dispatcher, throttles progress, streamed text and previews to about 15 UI updates a second,
decodes bitmaps off the UI thread and writes output files from the worker.

### Speech recognition

The built-in `ISpeechRecognizer` serves encoder-decoder models with a prompted decoder (Whisper and its derivatives, from Hugging Face
checkpoints, whisper.cpp `.bin` files or `.slm` packages); `AddSharplightTasks()` registers its factory. Its model description has an
`encoder` and a `decoder` component (graphs), a `tokenizer` component (the tokenizer as a JSON attachment) and a `feature-extractor`
component (log-mel configuration and the mel filter bank as an attachment); the decoder's canonical metadata carries the control
tokens (`SpeechKeys`). A request computes the log-mel features of the whole recording on the host, then per 30 s window runs the
encoder, detects the language when `SpeechRecognitionOptions.Language` is null, and decodes greedily (or with `Sampling`, or with a
beam search of `BeamSize` beams scored like transformers' `num_beams`) with Whisper's timestamp rules. Long recordings advance to the last complete timestamp segment of each window; `ConditionOnPreviousText`
prompts each window with the text so far, `Prompt` conditions the first. `Timestamps = TimestampGranularity.None` decodes without
timestamp tokens (one segment per window, as transformers does by default). Each `TranscriptionSegment` carries its token ids, mean log
probability and no-speech probability. Not yet: temperature fallback, word timestamps. `TranscriptionSegment.Speaker` names the speaker of models that diarize (VibeVoice-ASR: one pass over a long recording, segments with speaker numbers and start and end times, yielded as each is complete).

The same recognizer serves the other prompted encoder-decoders: Moonshine (raw-waveform encoder over variable-length windows, start token
only, no timestamps) and Canary (FastConformer encoder, a prompt template with source and target language; `Translate` targets English,
requests without `Language` are English). A second built-in implementation, registered by `AddSharplightTasks()` as well, serves models
whose decoder component configures a frame-aligned decoding (`decoding` "ctc", "rnnt" or "tdt"; Parakeet): per window (up to 30 s, no
padding) the encoder, then on the host CTC greedy or prefix beam search (`BeamSize`) or greedy RNN-T / TDT over the `predict` and `joint`
graphs. Both answer through the same `ISpeechRecognizer` with the same options, segments and streaming; the frame-aligned one fills
`TranscriptionSegment.Words` for `TimestampGranularity.Word` from the token rows. Which implementation a model gets follows from its
description (`ITaskClientFactory.CanCreate`), so applications, the HTTP endpoints and Studio's Transcribe page need no changes.

```csharp
var services = new ServiceCollection().AddSharplight()
    .AddSpeechRecognizer(o => { o.ModelPath = "ggml-small.bin"; o.Devices = [slot]; });   // or a HF directory / .slm
await using var provider = services.BuildServiceProvider();
var recognizer = provider.GetRequiredService<ISpeechRecognizer>();
var result = await recognizer.TranscribeAsync(WavFile.Read("talk.wav"), new SpeechRecognitionOptions { Language = "en" });
foreach (var segment in result.Segments)
{
    Console.WriteLine($"[{segment.Start:mm\\:ss\\.ff} - {segment.End:mm\\:ss\\.ff}] {segment.Text}");
}
```

The sample `samples/Sharplight.Transcribe` (`sharplight-transcribe model audio.wav [--language en] [--translate] [--beam-size 5] [--format srt]`) is this
in a command-line tool. Models of other speech families (audio LLMs) plug in as their own `ITaskClientFactory` for the
same kind; the feature extraction sits behind a small internal interface so a graph-based front end can replace the host one.

Vision models share host-side preprocessing in Core: `ImageProcessorConfig` (resize mode and filter, center crop, rescale, mean and std;
the neutral form of a model's processor configuration, stored in the `feature-extractor` component's configuration under
`imageProcessor`) and `ImageProcessor` (Pillow-exact antialiased resize, crop, normalize to planar floats, `Patchify`). Importers translate
the model's own processor file into the config; the image clients never see a format name. Models with a `vision-encoder` component
(graph `image_encode`: patches in, pooled embedding and optionally logits out) run through the built-in classifier / embedder
implementation (`ImageEncoderAdapter`: ViT, DINOv2, ...); `services.AddImageClassifier(...)` and `services.AddImageEmbedder(...)` register
configured models like every other kind. CLIP and SigLIP (adapter `ImageTextAdapter`, `vision-encoder` + `text-encoder` + `tokenizer` components, graphs `image_encode` and `text_encode`) add the text tower
to the same embedder: `EmbedTextAsync` embeds a prompt into the image space and `ClassifyZeroShotAsync` scores an image against candidate prompts (softmax for CLIP, sigmoid for SigLIP, with the model's learned scale and bias).
Object detectors (DETR: `DetrAdapter`; RT-DETR: `RtDetrAdapter`; graph `detect`: normalized `pixels` in, per-query `logits` and `boxes` out) run through `services.AddObjectDetector(...)`; the box decoding, thresholding and
non-maximum suppression are generic host-side utilities in Core (`BoundingBox`, `BoxUtilities.NonMaxSuppression`, `ObjectDetectionOptions`). `samples/Sharplight.Vision` (`sharplight-vision classify|detect|depth model image.png [--label prompt]... [-o out.png]`)
is this in a command-line tool (`depth` writes a grayscale PNG, `detect` draws the boxes). Depth Anything (`DepthEstimationAdapter`, graph `estimate_depth`) runs through `services.AddDepthEstimator(...)`. Per-pixel segmentation (SegFormer, `SegformerAdapter`, graph `segment_semantic`, kind `image-segmentation-semantic`) runs through `services.AddSemanticSegmenter(...)` / `ISemanticSegmenter.SegmentAsync(image)`: a label map at the image size, the class names, `Mask(classId)` and, for matting models, `Alpha`.

Promptable segmentation (SAM 2: `Sam2Adapter`, `vision-encoder` + `mask-decoder` components, graphs `segment_encode` and `segment_decode`) separates the expensive image stage from the cheap
prompt stage: `EncodeImageAsync` returns a `SegmentationImage` (embedding and skip features, held by the host), and every click or box of an interactive session is one `SegmentAsync` over it.
A prompt is points (positive or negative) and/or a `BoundingBox` in image pixels; the graph takes them as `point_coords` and `point_kinds` (`SegmentPromptKinds`) with a symbolic point count, so there is no limit
on the number of points. The decoder's low-resolution logits are resampled to the image on the host. OCR (TrOCR: the `encoder_decoder` adapter with an image encoder, `encoder.input` "image") reuses the speech recognizers' decoding machinery
(KV-cached decoder, cross-attention keys and values cached once per source) with the model's tokenizer; `ITextRecognizer` is the task client. Upscalers (Real-ESRGAN: `RrdbNetAdapter`, graph `upscale` of generic Conv, Resize and LeakyRelu ops)
are rebuilt for every input size through the model's dimensions (`height`, `width`); `IImageUpscaler` runs an image that fits one tile whole and cuts a larger one into overlapping tiles (`UpscaleOptions.TileSize`, `TilePadding`)
whose outputs are blended with linear weights, so seams are invisible. The official `RealESRGAN_x4plus.pth` loads through the restricted unpickler of `Sharplight.Import.PyTorch` (`SuperResolutionImporter`).
All three inference paths run on the thread pool, never on the caller's thread.

The importers and architecture adapters for speech and image models come in their own workstreams; the task layer defines the
interfaces, options, results, the description and the host-side plumbing they plug into.

### Audio codecs and voice activity

`audio-codec` models (EnCodec 24 kHz, DAC 44 kHz, Mimi; architectures `encodec`, `dac`, `mimi` in `Sharplight.Core/Architectures`, imported from Hugging Face
checkpoints by `Sharplight.Import.HuggingFace`) have an `encoder` component with the graph `audio_encode` (mono samples in, the code of every residual quantizer
stage out) and a `decoder` component with `audio_decode` (the codes of every stage and a codebook mask in, audio out); `AudioCodecGraphs` documents the
contract, so another codec is an adapter that emits the same graphs. `IAudioCodec.EncodeAsync` converts the audio to mono at the codec's rate, pads it to a whole number of
frames and runs it in segments of at most the component's frame bound; `AudioCodecOptions.Codebooks` keeps the first stages (the bit rate); `DecodeAsync` takes any
number of stages up to `CodebookCount` (the others are masked out). Time-domain graphs are built from `Conv1d` (causal reflect or zero padding, dilation, strides),
`ConvTranspose1d`, `Lstm`, the snake and ELU activations, residual quantization as `MatMul` + `ArgMax` + `Gather`, and Mimi's transformers (`Rope`, `Attention` with
a causal sliding window).

```csharp
using var codec = provider.GetRequiredService<TaskClientProvider>().Create<IAudioCodec>(model);
CodecCodes codes = await codec.EncodeAsync(WavFile.Read("talk.wav"), new AudioCodecOptions { Codebooks = 8 });
AudioBuffer again = await codec.DecodeAsync(codes);
```

`voice-activity` models (Silero VAD v5, imported from its 16 kHz ONNX export by `Sharplight.Import.SileroVad`) have one component whose stateful graph `vad_step` runs once
per 512-sample frame (the frame with the 64 samples before it, and the recurrent state in and out). `IVoiceActivityDetector.ProbabilitiesAsync` returns the frame
probabilities, `DetectAsync` the segments (`SpeechSegmentation.FromProbabilities` is the reference implementation's hysteresis, minimum durations, padding and splitting
of overlong segments, usable with any detector). Long-form speech recognition can use a detector as pre-segmentation:
`new SpeechRecognitionOptions { UseVoiceActivityDetection = true, VoiceActivityDetector = detector }` (or resolved from the container, see "Registration" below) transcribes only the speech, in windows of at most 30 s that start
at speech, with segment and word times on the recording's timeline.

`audio-embedding` models (CLAP, `laion/clap-htsat-unfused`; architecture `clap`) have an `encoder` component with the graph `audio_embed` (the log-mel features of a 10 s clip in, the projected
embedding out), a `text-encoder` component with `text_embed`, a tokenizer and a log-mel feature extractor (`AudioEmbedderGraphs` documents the contract). `IAudioEmbedder` embeds audio and text
into one space and scores audio against candidate texts (zero-shot tagging); audio shorter than a clip is repeated and padded, longer audio uses its first clip or the average of its clips
(`LongAudioMode`).

```csharp
using var embedder = provider.GetRequiredService<TaskClientProvider>().Create<IAudioEmbedder>(model);
var tags = await embedder.ClassifyZeroShotAsync(WavFile.Read("clip.wav"), ["a dog barking", "music", "speech"]);
Console.WriteLine($"{tags.Top!.Label}: {tags.Top.Score:P0}");
```

`music-generation` models (MusicGen; architecture `musicgen`) have a `text-encoder` (graph `text_encode`), a tokenizer, a `decoder` with the cached one-step graphs `decode` and
`decode_unconditional` (the two branches of classifier-free guidance; one code per codebook in, the logits of every codebook out) and a `vocoder` component with the codec's `audio_decode`
graph (`MusicGenerationGraphs`). `IMusicGenerator.GenerateAsync` returns the audio, `GenerateCodesAsync` the codec codes (the delay pattern removed), and `GenerateStreamingAsync` pieces
of audio while the codes are being generated (windowed codec decoding with context, so the pieces differ slightly from the one-shot audio at the borders).
`MusicGenerationRequest { Greedy = true }` is deterministic; otherwise top-k / top-p sampling with an optional seed.

```csharp
using var generator = clients.Create<IMusicGenerator>(model);
AudioBuffer music = await generator.GenerateAsync(new MusicGenerationRequest { Prompt = "calm piano with soft strings", Duration = TimeSpan.FromSeconds(10), Seed = 1 });
```

`audio-generation` models (ACE-Step 1.5 turbo; architecture `acestep`, imported by `Sharplight.Import.AceStep` from the release's folders) are latent-diffusion audio models: a tokenizer, a `text-encoder`
component (graphs `text_encode` and `lyric_embed`), an `encoder` component (`condition_encode`: lyrics, timbre reference and caption packed into the rows the denoiser attends to), a `denoiser`
(`denoise`: one evaluation of the diffusion transformer, the flow velocity) and a `vae` component (`audio_decode`: the Oobleck decoder, run in overlapping windows for long audio), see
`AudioGenerationGraphs`. `IAudioGenerator.GenerateAsync` returns 48 kHz stereo audio, `GenerateLatentsAsync` the denoised latents (25 Hz, 64 channels); `AudioGenerationRequest` has the prompt, optional
lyrics and language, duration, steps, shift, seed (or the initial noise itself) and guidance (ignored by the distilled model).

```csharp
using var generator = clients.Create<IAudioGenerator>(model);
AudioBuffer song = await generator.GenerateAsync(new AudioGenerationRequest { Prompt = "calm acoustic guitar with soft piano", Lyrics = "[Instrumental]", Duration = TimeSpan.FromSeconds(30), Seed = 1 });
```

Stable Audio Open 1.0 (architecture `stable_audio`, imported by `Sharplight.Import.Diffusers` from the diffusers `StableAudioPipeline` folders) is an `audio-generation` model with another graph contract,
`TimedAudioDiffusionGraphs`: a `text-encoder` (`text_encode`, T5), an `encoder` (`timing_encode`: the start and total seconds as two conditioning rows), a `denoiser` (`denoise`: one v-prediction of the
transformer on preconditioned latents) and the same `vae` graph (`audio_decode`). Its client samples with DPM-Solver++(2M) SDE and classifier-free guidance and returns 44.1 kHz stereo audio of up
to 47.5 s; the request's `NegativePrompt` steers the unconditional side. `IAudioGenerator` tells what a model takes: `SupportsLyrics`, `SupportsNegativePrompt`, `SupportsShift` and
`DefaultGuidanceScale` (null for a model without guidance).

```csharp
using var generator = clients.Create<IAudioGenerator>(model);   // stable-audio-open-1.0
AudioBuffer rain = await generator.GenerateAsync(new AudioGenerationRequest
{
    Prompt = "heavy rain on a tin roof, distant thunder", NegativePrompt = "music", Duration = TimeSpan.FromSeconds(20), Steps = 100, GuidanceScale = 7, Seed = 1,
});
```

Stable Audio 3 (architecture `stable_audio_3`, imported by `Sharplight.Import.StableAudio` from the stable-audio-3 checkpoint folders: `model_config.json`, `model.safetensors`, `t5gemma-b-b-ul2/`)
uses the rectified-flow form of `TimedAudioDiffusionGraphs` (the denoiser's `objective` is `rectified_flow` or `rf_denoiser`): a T5Gemma `text_encode`, a `timing_encode` of the total seconds, a
`denoise` velocity network whose latent length follows the duration, and a SAME autoencoder `audio_decode` over fixed windows. The base models sample with Euler steps and adaptive projected
guidance (50 steps, guidance 7 by default), the post-trained ones with ping-pong steps and no guidance (`DefaultGuidanceScale` null, no negative prompt). The SAME decoder adds a little random
dither as the reference package does; `AudioGenerationRequest.DecoderNoise` scales it (0 decodes deterministically).

```csharp
using var generator = clients.Create<IAudioGenerator>(model);   // stable-audio-3-small-music-base
AudioBuffer loop = await generator.GenerateAsync(new AudioGenerationRequest
{
    Prompt = "lofi house loop, 120 BPM", Duration = TimeSpan.FromSeconds(30), Steps = 50, GuidanceScale = 7, Seed = 1,
});
```

`audio-separation` models (Demucs `htdemucs`; architecture `demucs`) have one component whose graph `separate` runs one fixed-length segment (the host computes the spectrogram and the
normalizations and rebuilds the audio, see `SeparationGraphs`). `IAudioSeparator.SeparateAsync` converts the mixture to the model's rate and channels, runs overlapping segments with a
triangular cross-fade (`AudioSeparationOptions.Overlap`) and returns one `SeparatedStem` per source.

```csharp
using var separator = clients.Create<IAudioSeparator>(model);
IReadOnlyList<SeparatedStem> stems = await separator.SeparateAsync(WavFile.Read("song.wav"), progress: new Progress<double>(Console.WriteLine));
```

Registration with dependency injection: `AddSharplightTasks()` registers the factories of all audio kinds; `services.AddTaskClient<IVoiceActivityDetector>(options => ..., key)` registers a detector
once, and a speech recognizer resolves it from the container when a request sets `UseVoiceActivityDetection` (`SpeechRecognitionOptions.VoiceActivityDetectorKey` picks a keyed registration, otherwise
the unkeyed one; `SpeechRecognitionOptions.VoiceActivityDetector` still takes an instance directly).

### Image generation

`Sharplight.Import.Diffusers` reads Stable Diffusion 1.x / 2.x / XL checkpoints (diffusers folders, original single-file `.safetensors` or
`.ckpt` (through the restricted unpickler of Sharplight.Import.PyTorch),
stable-diffusion.cpp GGUF files) into the canonical `latent_diffusion` source; `LatentDiffusionAdapter` (Sharplight.Architectures)
builds a text encoder, a UNet denoiser and the VAE decoder / encoder as graphs of generic Core ops and describes them as an
`image-generation` model (components `text-encoder`, `tokenizer`, `denoiser`, `vae`, `scheduler`; SDXL's second text encoder is a
second graph of the `text-encoder` component whose pooled embedding and the size conditions feed the UNet). Core's built-in latent-diffusion
pipeline (`LatentDiffusionGeneratorFactory`, registered by `AddSharplightTasks()`) serves every model whose graphs follow
`DiffusionGraphs`; a family with another pipeline registers its own factory.

FLUX.1 (schnell, dev and fine-tunes) is the second family on the same pipeline: the importer translates diffusers folders,
original-name single files and transformer-only GGUF files (city96 / stable-diffusion.cpp) into the canonical `flux` source and
`FluxAdapter` builds the MMDiT transformer (dual- and single-stream blocks, 3-axis RoPE, modulation), the T5 encoder, the CLIP
encoder (pooled embedding only) and the 16-channel VAE from generic Core ops (components `text-encoder`, `text-encoder-2`,
`tokenizer`, `tokenizer-2`, `denoiser`, `vae`, `scheduler`). The scheduler component describes a rectified-flow schedule
(`predictionType: flow`, static or resolution-dependent shift), which the pipeline samples with `FlowMatchEulerSampler` (the
`euler` sampler of flow models); the transformer takes the pooled embedding and, for guidance-distilled models, the guidance scale
as inputs (`DiffusionGraphs.Guidance`), and the model turns classifier-free guidance off (`DiffusionConfigKeys.ClassifierFreeGuidance`).
The denoiser is built for one row; the pipeline runs a guided step row by row for such denoisers. The same contract fits the
other DiTs (SD3, Chroma, Qwen-Image; video DiTs add a frame axis to the latents).

Transformer-only checkpoints need companion files, given as auxiliary sources by role (`AuxiliaryRoles.TextEncoder` = CLIP-L,
`AuxiliaryRoles.SecondTextEncoder` = T5-XXL as GGUF or safetensors, `AuxiliaryRoles.Vae`); `FluxCompanionAdapter` merges them into
the canonical source, so the model is rebuilt for other sizes and saved as one `.slm` like a complete checkpoint. Roles the caller
leaves out are filled with the files the importer finds beside the transformer (`IModelImporter.FindCompanions`, also offered as
`ModelImporterCatalog.FindCompanions` for UIs):

```csharp
using var model = TaskModel.Load(loader, new TaskLoadOptions
{
    ModelPath = "flux1-schnell-Q8_0.gguf",
    Auxiliary = new Dictionary<string, string>
    {
        [AuxiliaryRoles.TextEncoder] = "clip_l.safetensors",
        [AuxiliaryRoles.SecondTextEncoder] = "t5-v1_1-xxl-encoder-Q8_0.gguf",
        [AuxiliaryRoles.Vae] = "ae.safetensors",
    },
    Devices = [slot],
});
using var generator = clients.Create<IImageGenerator>(model);
var result = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "a lighthouse at dusk" });   // 4 steps, 1024 × 1024
```

```csharp
var services = new ServiceCollection()
    .AddSharplight()
    .AddImageGenerator(o => { o.ModelPath = "v1-5-pruned-emaonly-Q8_0.gguf"; o.Devices = [slot]; });
await using var provider = services.BuildServiceProvider();
var generator = provider.GetRequiredService<IImageGenerator>();

var request = new ImageGenerationRequest { Prompt = "a lighthouse at dusk", NegativePrompt = "blurry" };
var options = new ImageGenerationOptions { Steps = 20, GuidanceScale = 7.5f, Seed = 42, Scheduler = Samplers.DpmPlusPlus2MKarras, PreviewInterval = 5 };
await foreach (var update in generator.GenerateStreamingAsync(request, options))
{
    Console.Write($"\r{update.Step}/{update.TotalSteps}");        // update.Preview: a latent-resolution preview every 5 steps
    if (update.Result is { } result)
    {
        PngCodec.Write("lighthouse.png", result.Images[0]);
    }
}
```

- Samplers (`Samplers`): `euler`, `euler-karras`, `euler-ancestral`, `ddim`, `pndm`, `dpm++2m`, `dpm++2m-karras` (flow models:
  `euler`, diffusers' `FlowMatchEulerDiscreteScheduler`); each follows the diffusers scheduler step by step and is configured from the model's `scheduler` component (`NoiseSchedule`: betas, prediction type,
  timestep spacing, offset; the original scheduler configuration is kept as the attachment `scheduler/source-config.json`). The
  default is the sampler the model was published with.
- Classifier-free guidance runs on the host: the negative prompt (or the empty prompt) and the prompt are one batched denoiser run
  per step (two runs for denoisers built for one row). Guidance-distilled models take the scale as a denoiser input instead, and
  models without either (FLUX.1-schnell) ignore it.
- The conditioning of the previous request is reused when the prompts are the same (new seeds, sizes or steps encode nothing).
- Memory: when the weights take more than 80 % of the device memory (FLUX's bf16 files on a 32 GB GPU), the generator holds either
  the text encoders or the denoiser and VAE: the text encoders are released after encoding, and the denoiser and VAE before new
  prompts are encoded (they are compiled again from the cached kernels). Image i of a request uses `Seed + i`; the noise is a counter-based Gaussian stream, identical on every machine and
  provider. `ImageGenerationOptions.InitialNoise` replaces it (reproducing another implementation's run).
- Image-to-image: `ImageGenerationRequest.Image` with `Strength`; inpainting: plus `Mask` (white = regenerate; the input is
  re-imposed outside the mask after every step).
- Two stages (the SDXL base with the SDXL refiner): `ImageGenerationOptions.Refiner` is a second generator that takes over the latents
  after `RefinerStart` (default 0.8) of the noise schedule (diffusers' `denoising_end` / `denoising_start`), denoises the rest on its own
  conditioning and decodes; the updates count the steps of both. Models conditioned on an aesthetic score (the refiner) take
  `AestheticScore` / `NegativeAestheticScore` (6 / 2.5).
- LoRA: LoRA files are auxiliary sources with roles starting with `AuxiliaryRoles.Lora` (`lora`, `lora:2`, ...; a `@scale` suffix sets
  the strength, e.g. `lora@0.8`). Their importer turns them into an `IWeightPatch` (`ModelSource.FormatData`) that `WeightPatches.Merge`
  applies to the primary's canonical weights before the adapter builds the model (W + s · α / r · up · down, computed when the weight
  is read; quantized weights become f16). The diffusers importer reads kohya-ss files (SD 1.x diffusers names, SDXL's SGM names,
  FLUX's original names with fused q/k/v), diffusers / PEFT files (`lora_A` / `lora_B`, also with fused targets split into row blocks)
  and ComfyUI's, for SD 1.x / 2.x / XL, FLUX, Chroma and Qwen-Image; LyCORIS / DoRA variants are rejected. A package saved with
  `sharplight-convert` keeps the merged weights.

```csharp
using var model = TaskModel.Load(loader, new TaskLoadOptions
{
    ModelPath = "sd_xl_base_1.0.safetensors",
    Auxiliary = new Dictionary<string, string> { [$"{AuxiliaryRoles.Lora}@0.8"] = "pixel-art-xl.safetensors" },
    Devices = [slot],
});
```
- Sizes: the graphs are built for the native size (512×512 for SD 1.x, 768×768 for SD 2.x 768-v, 1024×1024 for SDXL and FLUX) and stored
  that way in `.slm`; another `Width` / `Height` (multiples of 64; FLUX: of 16) rebuilds the model once per size through
  `TaskModel.DefinitionFor(dimensions)` (the adapter rebuilds from the canonical metadata and weights, so `.slm` packages work too).
- Classifier-free guidance with SDXL: without a negative prompt the unconditional row is all zeros (the model's
  `force_zeros_for_empty_prompt`); an explicit `NegativePrompt = ""` encodes the empty prompt.
- `sharplight-generate-image` (samples/Sharplight.GenerateImage) is a command-line front end for all of this.
- HTTP: `services.AddSharplightImageGenerator(name: "sd15")` serves the container's `IImageGenerator` on the OpenAI-compatible
  `POST /v1/images/generations` (`prompt`, `n`, `size`, `response_format: "b64_json"` with PNG data, plus the extensions
  `negative_prompt`, `steps`, `guidance_scale`, `seed` and `sampler`); `MapSharplight()` maps it when a generator is registered,
  `MapSharplightImages()` maps it alone. Requests to one generator run one at a time.

### Video generation

Video generators are models of kind `video-generation` with the client `IVideoGenerator` (`GenerateAsync(VideoGenerationRequest)`,
`GenerateStreamingAsync` → `VideoGenerationUpdate`), built like the image generators: `Sharplight.Import.Diffusers` reads Wan pipeline
folders (Wan 2.1 text-to-video, Wan 2.2 TI2V text- and image-to-video) into the canonical `wan` source, and `WanAdapter`
(Sharplight.Architectures) builds the umT5 text encoder (with a key mask), the video diffusion transformer (3-D patches, RoPE over
(frame, row, column), cross-attention to the text tokens, per-token timesteps for TI2V) and the causal 3-D VAE as graphs of generic
Core ops (components `text-encoder`, `tokenizer`, `transformer`, `vae`, `scheduler`; graph contract `VideoDiffusionGraphs`). LTX-Video
0.9.0 folders become the canonical `ltx` source (`LtxAdapter`: the unmasked T5 encoder, a transformer with key-masked
cross-attention and fractional RoPE, the non-causal VAE decoder over the whole latent video). Core's
built-in `VideoDiffusionGenerator` (registered by `AddSharplightTasks()`) runs every model that follows the contract.

```csharp
var services = new ServiceCollection()
    .AddSharplight()
    .AddVideoGenerator(o => { o.ModelPath = "Wan2.1-T2V-1.3B-Diffusers"; o.Devices = [slot]; });
await using var provider = services.BuildServiceProvider();
var generator = provider.GetRequiredService<IVideoGenerator>();

var request = new VideoGenerationRequest { Prompt = "a red fox trotting through snow", NegativePrompt = "blurry" };
var options = new VideoGenerationOptions { Width = 832, Height = 480, Frames = 33, Steps = 30, Seed = 7 };
await foreach (var update in generator.GenerateStreamingAsync(request, options))
{
    Console.Write($"\r{update.Phase} {update.Step}/{update.TotalSteps}");   // Denoising steps, then decoded frames (update.Frames)
    if (update.Result is { } video)
    {
        // video.Frames: 8-bit RGB frames, video.FramesPerSecond
    }
}
```

- Options (`VideoGenerationOptions`): width and height (multiples of the VAE's downscale times the patch size: 16 for Wan 2.1, 32
  for Wan 2.2 TI2V), frames (1 + 4k; other counts are rounded down), frames per second of the result, steps, guidance scale, seed,
  sampler (`unipc`, the Wan default, or `euler`: both follow the diffusers schedulers step by step with the model's flow shift),
  explicit initial latents [C, T, h, w] (`InitialNoise`) and preview frames. `VideoGenerationDefaults` (via `GetService`) reports the
  model's defaults (Wan 2.1: 832×480, 81 frames at 16 fps; Wan 2.2 TI2V: 1280×704, 121 frames at 24 fps; 50 steps, guidance 5)
  and whether it takes a start image.
- Text conditioning: the prompt is tokenized to 512 tokens, encoded with the padding masked out and the padding rows zeroed (the Wan
  pipelines' conditioning); classifier-free guidance runs the conditional and unconditional transformer one after the other.
- Image-to-video (`VideoGenerationRequest.Image`, models with first-frame conditioning such as Wan 2.2 TI2V): the start image is
  encoded by the VAE as the first latent frame, which replaces that frame in every transformer input (its tokens at timestep 0) and
  in the result.
- Decoding streams over the latent frames: the first latent frame decodes to one frame, each further one to four; the causal
  convolutions' last two input frames are carried from chunk to chunk (`CausalDecoderStream`), so memory does not grow with the
  clip length and `VideoGenerationUpdate.Frames` delivers the decoded frames as they come.
- Memory: when the weights take more than 40 % of the device memory (video activations are large), the text encoder is released
  after encoding and the transformer before decoding (compiled again from the cached kernels; the conditioning of the last prompts
  is reused, so the encoder only returns for new prompts).
- Transformer-only checkpoints (Wan GGUF files from city96 / QuantStack, original-name `.safetensors`): give a Wan diffusers folder of
  the same family as `Auxiliary[AuxiliaryRoles.TextEncoder]` and `Auxiliary[AuxiliaryRoles.Vae]` (`WanCompanionAdapter` merges its umT5
  encoder, tokenizer and VAE).
- `sharplight-generate-video` (samples/Sharplight.GenerateVideo) is a command-line front end.

## 1. Load a model (text generation)

The rest of this page covers text generation in depth: `LlmModel` and the serving stack around it. `LlmModel.Load` reads the file and builds the compute graphs. Nothing is compiled or uploaded to a device until you create a
session, a batch decoder or a vision encoder from the model. Dispose the model after everything created from it.

```csharp
using var model = LlmModel.Load(loader, "qwen3-4b-q4_k_m.gguf", load);   // loader: the container's ModelLoader
Console.WriteLine(model.Definition.Name);        // architecture, weights, metadata
string? template = model.ChatTemplate;           // the model's Jinja chat template, if it has one
```

`LlmLoadOptions` controls the devices (several devices split the layers; a CPU `Fallback` takes what does not fit), the context
length, the maximum rows per forward pass (`MaxBatchTokens`), disabled device features and auxiliary files such as a vision
projector (`Auxiliary = { [AuxiliaryRoles.Mmproj] = path }`). It extends `TaskLoadOptions` (the settings every model kind shares) and
has settable properties, so it binds from configuration like every options class.

## 2. Generate

The simplest path is one sequence: an `InferenceSession` holds the context (KV cache), a `Generator` runs the sampling loop.
Generation is a lazy iterator; successive calls continue the same context.

```csharp
using var session = model.CreateSession();
var generator = new Generator(session, model.Tokenizer);
var options = new GenerationOptions
{
    MaxNewTokens = 128,
    Sampling = new SamplingOptions { Temperature = 0.7f, TopP = 0.95f, Seed = 42 },
    StopStrings = ["\n\n"],
};
foreach (var token in generator.Generate("The capital of France is", options))
{
    Console.Write(token.Text);
}

Console.WriteLine($"{generator.LastStats!.DecodeTokensPerSecond:F1} tok/s");
```

Images: create the session on `WellKnown.DecodeMultimodalGraph`, pass a `VisionEncoder` to the generator and give the prompt
one image placeholder per image.

```csharp
using var vision = model.CreateVisionEncoder();
using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
var generator = new Generator(session, model.Tokenizer, vision);
var image = new ImageInput(width, height, rgbBytes);          // Sharplight.Vision
foreach (var token in generator.Generate(promptWithImagePlaceholder, [image], options)) { /* ... */ }
```

## 3. Serve many requests: the engine

`InferenceEngine` is a continuous-batching scheduler. It owns a batch decoder (several sequence slots advanced in one forward
pass), admits requests from any thread, shares cached prompt prefixes between them and streams results. Use it instead of
sessions whenever more than one caller is involved.

```csharp
using var engine = new InferenceEngine(model.CreateBatchDecoder(slots: 4), model.Tokenizer);

var request = new GenerationRequest
{
    PromptTokens = model.Tokenizer.Encode("Once upon a time", addSpecial: true, parseSpecial: false),
    MaxTokens = 64,
    StopStrings = ["."],
    Priority = 1,                            // higher runs first when requests queue
    Timeout = TimeSpan.FromSeconds(30),
};
await foreach (var chunk in engine.GenerateAsync(request))
{
    Console.Write(chunk.Text);               // the last chunk carries Finish and Usage
}

var snapshot = engine.GetMetrics();          // queue depth, slot use, TTFT / inter-token histograms, prefix-cache hits
```

Admission failures are exceptions with a machine-readable `Code` (`QueueFullException`, `PromptTooLongException`,
`EngineStoppedException`, all deriving from `EngineException`). `DrainAsync` stops admission and waits for running requests.

On top of the engine, `ModelService` is the chat layer: it renders messages and tools through the chat template, parses
reasoning blocks and tool calls out of the output (`ToolCallFormats`) and returns typed events.

```csharp
var service = new ModelService(engine, model.ChatTemplate, new ModelServiceOptions { ModelName = "qwen3" });
var result = await service.StartChat(new ChatRequest
{
    Messages = [new ChatTurn("system", "Answer briefly."), new ChatTurn("user", "Name three primary colours.")],
    MaxTokens = 128,
}).CompleteAsync();
Console.WriteLine(result.Choices[0].Content);          // also .Reasoning, .ToolCalls, .Usage
```

### Scheduling

Every engine step has a token budget (`EngineOptions.MaxBatchTokens`, default the decoder's batch size): decoding sequences
get their rows first, prompts fill the rest in chunks of at most `PrefillChunk` rows, so a long prompt never stalls running
streams for more than one step (chunked prefill). Queued requests are admitted by `Priority`, then by fair share: among
equal priorities the `Tenant` with the least recent usage (tokens, decaying with `FairShareHalfLife`) goes first. A
cancelled request (token or client disconnect) frees its slot at the next step boundary. `MinTokens` suppresses end-of-
generation and stop tokens until reached; `StopAtEog = false` generates through them (benchmarks).

### Structured outputs

A request can carry an `IOutputConstraint`; the engine asks it for the allowed tokens before every sample, masks the logits
(a packed `TokenMask` bitset, so a GPU sampler can take the same mask as an input) and reports the chosen token back.
`GrammarConstraint` implements it for a byte-level context-free `Grammar` built from GBNF (the llama.cpp format), a regular
expression or a JSON schema. Masks are computed by walking a precompiled vocabulary trie of the tokenizer in step with the
grammar; per-stack-top results are cached and shared between requests, so a JSON string body costs a few hundred
microseconds after its first use (151k-token Qwen3 vocabulary).

```csharp
var schema = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""";
var constraint = new GrammarConstraint(Grammar.FromJsonSchema(schema), model.Tokenizer);   // reuse across requests
var request = new GenerationRequest { PromptTokens = prompt, MaxTokens = 200, Constraint = constraint };
```

The chat layer does this for you: `ChatRequest.ResponseFormat = OutputFormat.JsonSchema(schema)` (also `JsonObject`,
`Regex`, `Gbnf`, `Choice`) constrains the answer while a reasoning block before it stays free, and tool calls are constrained
to the model's tool-call syntax and each tool's parameter schema when the call is forced (`ToolChoiceMode.Required` /
`Function`) or the tool is `Strict` (then the grammar starts at the call marker, plain answers stay free). Supported
JSON-schema subset: see `JsonSchemaCompiler` (types, properties/required in order, enums/consts, anyOf/oneOf/allOf, local
`$ref` incl. recursion, array bounds, string lengths, patterns, common formats).

### Speculative decoding

`EngineOptions.Speculator` proposes draft tokens that the target model verifies in one multi-row forward pass; a draft token
is kept while it equals what the target samples at that position. That is speculative rejection sampling for a point draft
(accept with probability p(draft), else sample from p without the draft), so greedy output is token-identical and sampled
output is distributed exactly as without speculation. Two speculators exist:

- `NgramSpeculator` drafts by prompt lookup (continuations of the last n-gram earlier in the context), which pays off for
  edits, code and quoting.
- `MtpSpeculator` (`LlmModel.CreateMtpSpeculator(slots, draftTokens)`, when `LlmModel.HasDraftHead`) runs the model's own
  multi-token-prediction head (the `mtp_draft` graph that adapters build from MTP / NextN layers, see `MtpDraftHead`): its
  own multi-sequence decoder with one attention layer and KV cache per MTP layer, fed the target's hidden states
  (`ISpeculator.UsesHiddenStates`, `Observe`; the decoder captures them through `IBatchDecoder.CaptureHiddenStates` /
  `HiddenStates`). Drafts are the head's greedy choices, chained k times per step. Heads: Qwen3.5 / Qwen3-Next MTP layers,
  the hyper-connected MTP head of Qwen3.8-Flash-Next (`qwen4exp`: reads the trunk's residual streams before the final mixer,
  the decode output `hidden_streams` named by `WellKnown.DraftHiddenSource`, and mixes its own streams before the LM head;
  `MtpDraftHead.BuildHyperConnected`), and the NextN layer of DeepSeek-V3 (`latent_decoder`), GLM-4.5 and MiMo-7B (`decoder`, any family whose checkpoint carries a NextN
  layer after uniform layers; `MtpDraftHead.FirstNextnModule`). The head compiles against `LlmModel.SharedWeights`, so its
  embedding and LM head are the decoder's buffers (only its own layer adds device memory).

Verification runs on the device when the decoder samples there (`LlmLoadOptions.DeviceSampling`, `IDeviceSamplingDecoder`):
every verification row is sampled (greedy or from the sequence's seeded stream at that row's position) by the sampling head,
and the step reads back one token and log-probability per row instead of logits rows. The decoder downloads hidden states
only for the rows the draft head asks for (`HiddenStates(input, rows)`), and not at all when nothing captures them.

The decoder must be created with room for the verification rows and the drafts
(`CreateBatchDecoder(slots, maxLogitsRows: ..., draftTokens: k)`; `ChatModelOptions.Speculator` / `MtpDraftTokens` do that
for hosted models). Attention-only models roll back any number of rows by truncation; models with recurrent state (gated
delta nets, Mamba, RWKV, short convs) get their recurrent ops compiled in rollback form for `k + 1` rows
(`Graph.RecurrentRollback`, `IBatchDecoder.MaxSpeculativeRows`): a speculative run records its per-row op inputs and leaves
the state at the run start, `Truncate` picks how many rows to keep, and the slot's next run replays them inside the same
kernel before its own rows. The engine speculates while at most `SpeculationMaxBatch` sequences decode and verifies at most
`SpeculationMaxDraftRows` drafts per step (fewer per sequence as the batch grows); requests with an output constraint are not
speculated. CLI and server: `--speculate-mtp k`, `--speculate-ngram k` (the CLI's `--gpus N` places the model over N physical GPUs; `--draft <file>` loads an MTP draft file such as
`mtp-*.gguf`, an EAGLE-3 or a DFlash drafter as the `draft` auxiliary source; Studio: the load option "Draft head tokens", 0 = off); `sharplight-convert precompile --drafts k` stores
the kernels of the verifying decoders and the draft head. `LlmModel.EstimateFootprint(slots, ctx, kvCacheTokens, draftTokens,
draftHead, maxLogitsRows)` counts the rollback record states, the verification logits rows and (with `draftHead`) the head's
own weights, KV cache and activations; the hosted loader uses it for admission when `MtpDraftTokens` is set.

### Weight sharing between compiled models

`CompileOptions.SharedWeights` (a `SharedWeights` registry) lets compiled models of the same `ModelSource` use one device
buffer per weight: buffers are keyed by device, weight source, type, size and the provider's layout (repack), reference
counted, filled by the first holder and freed with the last. Providers opt in through `IComputeDevice.SharesWeights` (CPU and
Vulkan do); placement does not count weights already resident. `LlmModel` passes its registry (`LlmModel.SharedWeights`) to
every decoder and draft head it compiles.

### Several models

`ModelHost` registers models by name, loads them on first use within the device memory budget, evicts the least recently used
idle model when memory is short, unloads idle ones after a timeout and hands out leases so a model in use is never unloaded.

```csharp
using var host = new ModelHost(new ModelHostOptions { ReserveBytes = 2L << 30, IdleTimeout = TimeSpan.FromMinutes(10) });
host.Register(new ModelRegistration
{
    Name = "qwen3-4b",
    Loader = new LlmChatModelLoader(loader, "qwen3-4b", "qwen3-4b-q4_k_m.gguf", load, new ChatModelOptions { Slots = 4 }),
});

using var lease = await host.AcquireAsync("qwen3-4b");            // loads if needed
var chat = lease.As<ChatModelInstance>().Service;
```

## 4. Serve over HTTP

`Sharplight.AspNetCore` maps OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) and Anthropic-compatible
(`/v1/messages`, `/v1/messages/count_tokens`) endpoints, `/v1/embeddings` and `/v1/rerank` for registered retrieval models, `/v1/audio/transcriptions` and `/v1/audio/translations` for
registered speech recognizers (`services.AddSharplightSpeechRecognizer("whisper-1")` serves the container's `ISpeechRecognizer`; multipart
WAV upload, `json`, `text`, `srt`, `vtt` and `verbose_json` responses),
`/health` (summary), `/health/live` and `/health/ready` (probes; ready answers 503 while draining or loading), `/metrics`
(Prometheus: queue depth, batch size, step time, TTFT / inter-token histograms, KV-cache use, speculation acceptance) and
`/admin/models`. Beyond the standard fields the APIs accept `ignore_eos`, `min_tokens`, `priority`, `user` (the fair-share
tenant; else the API key), `response_format` (`json_object`, `json_schema`), strict tools (`tools[].function.strict`), the
vLLM guided-decoding fields (`guided_json`, `guided_regex`, `guided_grammar`, `guided_choice`, `structured_outputs`) and,
on `/v1/messages`, `output_format` and `metadata.user_id`. A full queue answers 429 with `Retry-After`.

```csharp
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSharplightServer(host, o =>               // a ModelHost; or pass a ModelService for a single model
{
    o.ApiKeys.Add("secret");
    o.DefaultSampling = new SamplingParams { Temperature = 0.7f, TopP = 0.95f };
});

var app = builder.Build();
app.UseSharplightServer();      // CORS, API keys, error mapping; call before the Map* methods
app.MapSharplightServer();      // everything the options enable
app.Run();
```

`SharplightServerOptions` switches the APIs on and off, sets keys, CORS origins, request limits and timeouts, and decides
whether an unknown `model` falls back to the default one. Existing OpenAI / Anthropic clients only change their base URL.

With Sharplight.Hosting, `services.AddSharplightHosting(...)` plus `services.AddSharplightServer(o => ...)` serves the container's
`ModelHost` (chat models, and embedding / reranker models of the host on `/v1/embeddings` and `/v1/rerank`); `GET /admin/status`
returns the hosting status. Multi-tenant hosts map the APIs into route groups with their own authentication, per-request resolver
(models and aliases, defaults, allowed tools, isolation scope), admission hooks and usage reporting: see
[serving-multitenant.md](serving-multitenant.md).

## 5. Embeddings and reranking

`EmbeddingModel` runs encoder embedders (BERT family, bge, e5, MiniLM, nomic) and decoder embedders (Qwen3-Embedding).
Inputs are tokenized, truncated, packed and run in batches; pooling and normalization follow the model's metadata.

```csharp
using var embedder = EmbeddingModel.Load(loader, "bge-small-en-v1.5", new EmbeddingLoadOptions { Devices = [slot] });

EmbeddingResult result = embedder.Embed(["first text", "second text"]);
float[] vector = result.Vectors[0];                       // embedder.Dimensions long, L2-normalized when the model says so
float[] query = embedder.Embed("what is a vector?", new EmbedOptions { Prompt = "query" });   // a prompt the model defines
float[] small = embedder.Embed("text", new EmbedOptions { Dimensions = 256 });                // Matryoshka cut
```

`Reranker` scores documents against a query with a cross-encoder or a yes/no decoder reranker:

```csharp
using var reranker = Reranker.Load(loader, "bge-reranker-v2-m3", new RerankerLoadOptions { Devices = [slot] });
RerankResult scores = reranker.Score("what is a vector?", ["a quantity with direction", "a kind of animal"]);
// scores.Scores in [0, 1] per document, in input order
```

Both implement small interfaces (`ITextEmbedder`, `ITextReranker`) that the server and the Microsoft.Extensions.AI adapter
depend on, so a test double or a remote implementation can stand in.

## 6. Microsoft.Extensions.AI

`SharplightChatClient` implements `IChatClient`: messages and tools are rendered with the model's chat template, reasoning
becomes `TextReasoningContent`, tool calls become `FunctionCallContent` (wrap with `UseFunctionInvocation()` to run them) and
images are `DataContent`. `SharplightEmbeddingGenerator` implements `IEmbeddingGenerator<string, Embedding<float>>`.

```csharp
var services = new ServiceCollection().AddSharplight();
services.AddSharplightChatClient(o => { o.ModelPath = "qwen3-4b-q4_k_m.gguf"; o.Devices = [slot]; }, client => client.EnableThinking = false)
    .UseFunctionInvocation();
services.AddSharplightEmbeddingGenerator(o => { o.ModelPath = "bge-small-en-v1.5"; o.Devices = [slot]; });

var provider = services.BuildServiceProvider();
var chat = provider.GetRequiredService<IChatClient>();
var answer = await chat.GetResponseAsync("Why is the sky blue?");
Console.WriteLine(answer.Text);
```

The default backend runs one generation at a time. To let many concurrent callers share a model, implement
`ITextGenerationBackend` over an `InferenceEngine` and pass it to the `SharplightChatClient` constructor. See also
[extensions-ai.md](extensions-ai.md).

The other task kinds map the same way: `SharplightSpeechToTextClient` (`ISpeechToTextClient` over an `ISpeechRecognizer`, WAVE input),
`SharplightTextToSpeechClient` (`ITextToSpeechClient` over an `ISpeechSynthesizer`, WAVE output) and `SharplightImageGenerator`
(`IImageGenerator` over a Sharplight image generator, PNG output). `services.AddSharplightSpeechToTextClient(key)`,
`.AddSharplightTextToSpeechClient(key)` and `.AddSharplightImageGenerator(key)` wrap the container's task clients.

## 7. Precompilation: portable code and device code for every provider

Every execution provider implements a two-stage code generation contract on `IExecutionProvider`; the CPU returns empty sets, so
hosts treat all providers alike (details and the Metal mapping: [model-format.md](model-format.md#two-stage-precompilation)).

```csharp
using var services = SharplightDefaults.CreateServiceProvider();
foreach (var (provider, availability) in services.GetRequiredService<ExecutionProviderCatalog>().Discover().Where(p => p.Availability.IsAvailable))
{
    foreach (var info in provider.GetDevices())
    {
        using var device = provider.OpenDevice(info);
        // Stage 1: portable code (SPIR-V for Vulkan) for the device's profile; needs no driver, works for absent devices too.
        PortableCodeSet portable = await provider.CompilePortableAsync(new PortableCompilationRequest(device.CreateProfile(), workload), ct);
        // Stage 2: the driver's compiled form for exactly this device and driver (Vulkan: pipeline binaries or a pipeline cache).
        DeviceCodeSet binaries = await provider.CompileDeviceSpecificAsync(device, portable, ct);
        ModelPackage.AddArtifacts(path, [portable.ToArtifactSet(), binaries.ToArtifactSet()]);
    }
}
```

`workload` (a `CompilationWorkload`) compiles the model's graphs on the device it is given, recording into the
`CompilationArtifacts` it is given. For packages, `PackagePrecompiler.PrecompileAsync(path, providers, options, progress, ct)` does
all of this: the loaders' graphs as the workload, identical devices sharing their sets, a `PackagePrecompileReport`, and one
crash-safe in-place write. Loads pick the best stored stage per kernel on their own (device binary, else portable code, else
compile), and validate device code before the driver sees it. After a load, `CompilationArtifacts.CompileDeviceCodeAsync(providers)`
turns what it compiled into device code (`LlmModel.Artifacts`, `TaskModel.Artifacts`).

The same artifact sets also live in a second place: the per-user `ArtifactStore` (`artifacts` in the cache folder of the storage
layout, `ArtifactStore.DirectoryFor(layout)`; see [storage.md](storage.md)). It uses the descriptors, set ids, keys, tiers and matching of the sets in a package, one subdirectory per set with
its descriptor in `descriptor.json`. Loads look in the package's sets first, then in the store, and only then compile. New SPIR-V
is written to the store, and so is the driver's pipeline cache, in the pipeline-cache set of the device and driver. Packages change
only when a host precompiles or converts on purpose. The store helps whenever a package cannot: GGUF, Hugging Face and ONNX models
loaded straight from their files, graphs built in code (tests), and kernel development (entries are content addressed, so only
changed kernels compile again). Device code a host made for such a model can be kept there with `ArtifactStore.Add(deviceCodeSet)`.
Every entry file holds its key and a SHA-256, and every descriptor must hash to its directory name, so damaged, foreign or stale
entries are ignored rather than trusted. Writes are atomic, several processes can share the store, and it is size capped
(least recently used entries are evicted past 2 GiB). `SHARPLIGHT_COMPILATION_CACHE=<dir>` moves the store, `=off` turns it off,
and per device `DeviceOptions.KernelCacheDirectory` picks the directory and `DeviceOptions.UseCompilationCache = false` turns it
off.

```csharp
var options = new DeviceOptions { KernelCacheDirectory = "/data/sharplight-artifacts" };  // or leave it null: the storage layout's store
using var device = provider.OpenDevice(info, options);
ArtifactStore? store = device.ArtifactStore;                // the store this device reads and records into
foreach (var set in store?.ArtifactSets ?? [])
{
    Console.WriteLine(ArtifactDescriptor.Format(set.Descriptor));
}
```

## Conventions

- Types that own device memory or threads (`LlmModel`, `TaskModel`, every task client, `InferenceSession`, `InferenceEngine`,
  `ModelHost`, `EmbeddingModel`, `Reranker`, `VisionEncoder`) are `IDisposable`; dispose them in reverse order of creation (a
  service container does that for the clients and models it created).
- `InferenceEngine`, `ModelHost`, `ModelService`, `EmbeddingModel` and `Reranker` are thread-safe. Sessions, generators and
  samplers are not: one thread at a time.
- Request options are immutable records with defaults that work. Load and host options are classes with settable properties for
  Microsoft.Extensions.Options: `TaskLoadOptions` and its kind-specific subclasses `LlmLoadOptions`, `EmbeddingLoadOptions`,
  `RerankerLoadOptions`, `SpanScorerLoadOptions`, plus `AdapterOptions`, the provider and importer options and `SharplightServerOptions`.
  Only `Devices` is required (checked when the model is compiled or loaded).
- Failures that a caller can act on are typed exceptions: `EngineException` and its subclasses for admission, `ModelHostException`
  and its subclasses for hosting, `DeviceOutOfMemoryException` (an `OutOfMemoryException`) when a device has no memory left after
  every fallback.
- No provider assumes a memory topology or size: memory types, staging sizes and blocks adapt to the device, and
  `DeviceOptions.MemoryLimits` (or `SHARPLIGHT_LIMIT_*`) simulates smaller hardware; see [hardware.md](hardware.md) for each
  provider's requirements and fallbacks.
- The CPU provider's kernels do not depend on the application's JIT settings. By default .NET first compiles a method with loops as
  unoptimized tier-0 code and switches to optimized code only after on-stack replacement and tiering catch up, so the first runs of a
  model (and short benchmarks) would measure unoptimized kernels. Every method with a loop in `Sharplight.Cpu` is marked
  `MethodImplOptions.AggressiveOptimization` and compiled fully optimized on first use, whatever `TieredCompilationQuickJitForLoops`
  the application sets. In an application with the .NET defaults, runs 1-5 of resnet50 on the CPU averaged about 48 ms without the
  attribute and 10 ms with it.
