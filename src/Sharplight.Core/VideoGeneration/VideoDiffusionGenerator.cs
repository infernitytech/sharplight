using System.Net;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Sharplight.ImageGeneration;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.VideoGeneration;

/// <summary>
/// The built-in video diffusion pipeline: runs any video-generation model whose description follows <see cref="VideoDiffusionGraphs"/>
/// (a masked text encoder, a video transformer, a causal VAE decoder streamed over latent frames, tokenizer and flow scheduler
/// components). The prompt and the negative prompt are encoded once (the padding rows of the conditioning zeroed, as the Wan
/// pipelines do); the seeded noise (or <see cref="VideoGenerationOptions.InitialNoise"/>) is denoised by the sampler with
/// classifier-free guidance on the host (the conditional and the unconditional run one after the other), then decoded latent frame
/// by latent frame, the causal convolutions' caches carried from chunk to chunk. Image-to-video with first-frame conditioning (Wan
/// 2.2 TI2V) encodes the start image as the first latent frame, which replaces that frame in every model input and in the result,
/// with timestep 0 for its tokens. Graphs are compiled on first use, per size.
/// </summary>
internal sealed partial class VideoDiffusionGenerator : IVideoGenerator
{
    /// <summary>Weights above this fraction of the device memory make the generator hold one stage at a time.</summary>
    private const double HoldOneStageFraction = 0.4;

    private readonly TaskModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly NoiseSchedule _schedule;
    private readonly int _contextLength;
    private readonly int _latentChannels;
    private readonly int _spatialDownscale;
    private readonly int _temporalDownscale;
    private readonly int _patchSize;
    private readonly bool _tokenTimesteps;
    private readonly bool _firstFrameConditioning;
    private readonly int _defaultWidth;
    private readonly int _defaultHeight;
    private readonly int _defaultFrames;
    private readonly double _framesPerSecond;
    private readonly int _defaultSteps;
    private readonly float _defaultGuidance;
    private readonly float _defaultGuidance2;
    private readonly double? _boundaryTimestep;
    private readonly string? _defaultNegative;
    private readonly float[] _rgbFactors;
    private readonly bool _holdOneStage;
    private readonly bool _wholeVideoDecoder;
    private readonly Dictionary<string, CompiledModel> _compiled = new(StringComparer.Ordinal);
    private (string Prompt, string? Negative, bool Guided, PromptRow[] Rows)? _lastConditioning;
    private bool _disposed;

    /// <summary>Creates the pipeline for a model (the model stays owned by the caller).</summary>
    /// <param name="model">A video-generation model following <see cref="VideoDiffusionGraphs"/>.</param>
    public VideoDiffusionGenerator(TaskModel model)
    {
        _model = model;
        var description = model.Description;
        _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        _schedule = NoiseSchedule.FromComponent(description.Component(ModelComponentRoles.Scheduler));
        _contextLength = description.Component(ModelComponentRoles.TextEncoder).ConfigValue(DiffusionConfigKeys.ContextLength, 512);
        var transformer = description.Component(ModelComponentRoles.Transformer);
        _patchSize = transformer.ConfigValue(DiffusionConfigKeys.PatchSize, 2);
        _tokenTimesteps = transformer.ConfigValue(VideoConfigKeys.TokenTimesteps, false);
        var vae = description.Component(ModelComponentRoles.Vae);
        _latentChannels = vae.ConfigValue(DiffusionConfigKeys.LatentChannels, 16);
        _spatialDownscale = vae.ConfigValue(DiffusionConfigKeys.Downscale, 8);
        _temporalDownscale = vae.ConfigValue(VideoConfigKeys.TemporalDownscale, 4);
        _rgbFactors = vae.Config.TryGetValue(DiffusionConfigKeys.LatentRgbFactors, out var factors)
            ? [.. factors.EnumerateArray().Select(f => f.GetSingle())]
            : [];
        _wholeVideoDecoder = vae.Graphs.Contains(VideoDiffusionGraphs.VaeDecode);
        _firstFrameConditioning = description.ConfigValue(VideoConfigKeys.FirstFrameConditioning, false) && vae.Graphs.Contains(VideoDiffusionGraphs.VaeEncode);
        _defaultWidth = description.ConfigValue(DiffusionConfigKeys.Width, 832);
        _defaultHeight = description.ConfigValue(DiffusionConfigKeys.Height, 480);
        _defaultFrames = description.ConfigValue(VideoConfigKeys.Frames, 81);
        _framesPerSecond = description.ConfigValue(VideoConfigKeys.FramesPerSecond, 16.0);
        _defaultSteps = description.ConfigValue(DiffusionConfigKeys.Steps, 50);
        _defaultGuidance = description.ConfigValue(DiffusionConfigKeys.GuidanceScale, 5f);
        _defaultGuidance2 = description.ConfigValue(VideoConfigKeys.GuidanceScale2, _defaultGuidance);
        _boundaryTimestep = description.TryGetComponent(VideoDiffusionGraphs.SecondTransformerRole, out _)
            ? description.ConfigValue(VideoConfigKeys.BoundaryRatio, 0.875) * _schedule.TrainSteps
            : null;
        _defaultNegative = description.Config.TryGetValue(VideoConfigKeys.NegativePrompt, out var negative) ? negative.GetString() : null;
        // Video activations are large (the VAE decoder at full resolution, attention over tens of thousands of tokens): unless the
        // weights take little of the device, hold either the text encoder or the transformer and VAE (compiled again from the
        // cached kernels; the conditioning of the last prompts is reused, so the encoder only returns for new prompts).
        long weightBytes = model.Definition.Weights.Values.Sum(w => w.ByteSize);
        long deviceBytes = model.Options.Devices.Sum(slot => slot.Device.Info.TotalMemory);
        _holdOneStage = deviceBytes > 0 && weightBytes > deviceBytes * HoldOneStageFraction;
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    private int SizeMultiple => _spatialDownscale * _patchSize;

    /// <inheritdoc/>
    /// <remarks>Also answers <see cref="VideoGenerationDefaults"/>.</remarks>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is null && serviceType == typeof(VideoGenerationDefaults))
        {
            return new VideoGenerationDefaults(_defaultWidth, _defaultHeight, _defaultFrames, _framesPerSecond, _defaultSteps, _defaultGuidance, _schedule.Sampler,
                SizeMultiple, _temporalDownscale, _firstFrameConditioning, _defaultNegative);
        }

        return serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model
            : serviceKey is null && serviceType.IsInstanceOfType(this) ? this
            : null;
    }

    /// <inheritdoc/>
    public async Task<VideoGenerationResult> GenerateAsync(VideoGenerationRequest request, VideoGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        VideoGenerationResult? result = null;
        await foreach (var update in GenerateStreamingAsync(request, options, cancellationToken).ConfigureAwait(false))
        {
            result = update.Result ?? result;
        }

        return result ?? throw new InvalidOperationException("Video generation produced no result.");
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoGenerationUpdate> GenerateStreamingAsync(VideoGenerationRequest request, VideoGenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (request.Image is not null && !_firstFrameConditioning)
        {
            throw new NotSupportedException($"{_model.Definition.Name} generates from text only; it does not take a start image.");
        }

        var run = Plan(options ?? VideoGenerationOptions.Default);
        yield return new VideoGenerationUpdate(VideoGenerationPhase.Conditioning, 0, 1);
        string? negative = request.NegativePrompt ?? _defaultNegative;
        var conditioning = await Task.Run(() => Conditioning(request.Prompt, negative, run.Guided), cancellationToken).ConfigureAwait(false);
        var firstFrame = request.Image is null ? null : await Task.Run(() => EncodeImage(request.Image, run), cancellationToken).ConfigureAwait(false);
        yield return new VideoGenerationUpdate(VideoGenerationPhase.Conditioning, 1, 1);

        var sampler = Samplers.Create(run.Sampler, _schedule, run.Steps, TokenCount(run));
        var latents = run.InitialNoise is { } explicitNoise ? explicitNoise.AsSpan(0, run.LatentSize).ToArray() : LatentNoise.Draw(run.Seed, 0, run.LatentSize);
        for (int i = 0; i < latents.Length; i++)
        {
            latents[i] *= sampler.InitialNoiseScale;
        }

        int evaluations = sampler.Timesteps.Count;
        long noiseStreams = 1;
        for (int index = 0; index < evaluations; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int step = index;
            var prediction = await Task.Run(() => Denoise(run, conditioning, sampler, step, latents, firstFrame), cancellationToken).ConfigureAwait(false);
            long noiseStream = noiseStreams++;
            sampler.Step(index, prediction, latents, () => LatentNoise.Draw(run.Seed, noiseStream, latents.Length));
            bool preview = run.PreviewInterval > 0 && _rgbFactors.Length > 0 && (index + 1) % run.PreviewInterval == 0;
            yield return new VideoGenerationUpdate(VideoGenerationPhase.Denoising, index + 1, evaluations,
                Preview: preview ? Preview(sampler.PredictedOriginal ?? latents, run, firstFrame) : null);
        }

        if (firstFrame is not null)
        {
            ReplaceFirstFrame(latents, firstFrame, run);
        }

        if (_holdOneStage)
        {
            Release(CompiledKey(VideoDiffusionGraphs.Denoise, run.Dimensions));
            Release(CompiledKey(VideoDiffusionGraphs.Denoise2, run.Dimensions));
        }

        var frames = new List<ImageInput>(run.Frames);
        if (_wholeVideoDecoder)
        {
            var decoded = await Task.Run(() => DecodeWhole(run, latents), cancellationToken).ConfigureAwait(false);
            frames.AddRange(decoded);
            yield return new VideoGenerationUpdate(VideoGenerationPhase.Decoding, frames.Count, run.Frames, Frames: decoded);
        }
        else
        {
            var stream = new CausalDecoderStream();
            for (int latentFrame = 0; latentFrame < run.LatentFrames; latentFrame++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int current = latentFrame;
                var decoded = await Task.Run(() => stream.DecodeImages(Decoder(run, current), LatentFrame(latents, current, run), run.Width, run.Height),
                    cancellationToken).ConfigureAwait(false);
                frames.AddRange(decoded);
                if (latentFrame == 0)
                {
                    // The first-frame decoder is not needed again for this video; its activations make room for the next-frame decoder.
                    Release(CompiledKey(VideoDiffusionGraphs.VaeDecodeFirst, run.Dimensions));
                }

                yield return new VideoGenerationUpdate(VideoGenerationPhase.Decoding, frames.Count, run.Frames, Frames: decoded);
            }
        }

        var result = new VideoGenerationResult(frames, run.FramesPerSecond, run.Seed);
        yield return new VideoGenerationUpdate(VideoGenerationPhase.Completed, run.Frames, run.Frames, result);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var compiled in _compiled.Values)
        {
            compiled.Dispose();
        }

        _compiled.Clear();
    }

    private VideoRun Plan(VideoGenerationOptions options)
    {
        int multiple = SizeMultiple;
        int width = Math.Max(multiple, (options.Width ?? _defaultWidth) / multiple * multiple);
        int height = Math.Max(multiple, (options.Height ?? _defaultHeight) / multiple * multiple);
        int frames = options.Frames ?? _defaultFrames;
        ArgumentOutOfRangeException.ThrowIfLessThan(frames, 1, nameof(options));
        int latentFrames = (frames - 1) / _temporalDownscale + 1;
        int steps = options.Steps ?? _defaultSteps;
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1, nameof(options));
        float guidance = options.GuidanceScale ?? _defaultGuidance;
        int defaultLatentFrames = (_defaultFrames - 1) / _temporalDownscale + 1;
        bool native = width == _defaultWidth && height == _defaultHeight && latentFrames == defaultLatentFrames;
        var run = new VideoRun
        {
            Width = width,
            Height = height,
            Frames = 1 + (latentFrames - 1) * _temporalDownscale,
            FramesPerSecond = options.FramesPerSecond ?? _framesPerSecond,
            LatentChannels = _latentChannels,
            LatentFrames = latentFrames,
            LatentHeight = height / _spatialDownscale,
            LatentWidth = width / _spatialDownscale,
            Steps = steps,
            Guidance = guidance,
            Guidance2 = options.GuidanceScale2 ?? options.GuidanceScale ?? _defaultGuidance2,
            Guided = guidance > 1f,
            Seed = options.Seed ?? Random.Shared.NextInt64(1L << 53),
            Sampler = options.Scheduler ?? _schedule.Sampler,
            PreviewInterval = options.PreviewInterval,
            InitialNoise = options.InitialNoise,
            Dimensions = native
                ? null
                : new Dictionary<string, long>
                {
                    [VideoDiffusionGraphs.FramesDimension] = latentFrames,
                    [VideoDiffusionGraphs.Height] = height,
                    [VideoDiffusionGraphs.Width] = width,
                },
        };
        if (run.InitialNoise is not null && run.InitialNoise.Length < run.LatentSize)
        {
            throw new ArgumentException($"InitialNoise has {run.InitialNoise.Length} values; the latent video needs {run.LatentSize}.", nameof(options));
        }

        return run;
    }

    private int TokenCount(VideoRun run) => run.LatentFrames * (run.LatentHeight / _patchSize) * (run.LatentWidth / _patchSize);

    /// <summary>
    /// The conditioning rows (the prompt's, then the negative prompt's when guided), reused when the prompts are those of the previous
    /// request. Models that barely fit release the transformer and VAE before encoding and the text encoder after it.
    /// </summary>
    private PromptRow[] Conditioning(string prompt, string? negative, bool guided)
    {
        if (_lastConditioning is { } last && last.Prompt == prompt && last.Negative == negative && last.Guided == guided)
        {
            return last.Rows;
        }

        if (_holdOneStage)
        {
            foreach (string key in _compiled.Keys.Where(k => k != VideoDiffusionGraphs.TextEncode).ToList())
            {
                Release(key);
            }
        }

        PromptRow[] rows = guided ? [EncodePrompt(prompt), EncodePrompt(negative ?? "")] : [EncodePrompt(prompt)];
        if (_holdOneStage)
        {
            Release(VideoDiffusionGraphs.TextEncode);
        }

        _lastConditioning = (prompt, negative, guided, rows);
        return rows;
    }

    /// <summary>
    /// One prompt's conditioning: with a masked encoder its hidden states with the padding rows zeroed (Wan), otherwise the hidden states
    /// of all rows, the padding masked in the denoiser's cross-attention (LTX-Video).
    /// </summary>
    private PromptRow EncodePrompt(string prompt)
    {
        var tokens = PromptTokenizer.Tokenize(_tokenizer, _contextLength, CleanPrompt(prompt), out int count);
        var mask = new float[tokens.Length];
        mask.AsSpan(0, count).Fill(1f);
        var encoder = Compiled(ModelComponentRoles.TextEncoder, VideoDiffusionGraphs.TextEncode, null);
        encoder.SetInput<int>(Input(encoder, VideoDiffusionGraphs.Tokens), tokens);
        bool masked = encoder.Graph.Inputs.Any(i => i.Name == VideoDiffusionGraphs.TokenMask);
        if (masked)
        {
            encoder.SetInput<float>(Input(encoder, VideoDiffusionGraphs.TokenMask), mask);
        }

        var parameters = new RunParameters();
        encoder.Run(parameters);
        var hidden = new float[encoder.OutputShape(VideoDiffusionGraphs.Hidden, parameters).Aggregate(1L, (a, b) => a * b)];
        encoder.GetOutput<float>(VideoDiffusionGraphs.Hidden, hidden);
        if (masked)
        {
            int width = hidden.Length / tokens.Length;
            hidden.AsSpan(count * width).Clear();
        }

        return new PromptRow(hidden, mask);
    }

    /// <summary>The Wan pipelines' prompt cleaning: HTML entities decoded (twice), whitespace runs collapsed, trimmed.</summary>
    private static string CleanPrompt(string prompt)
        => Whitespace().Replace(WebUtility.HtmlDecode(WebUtility.HtmlDecode(prompt)).Trim(), " ").Trim();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();

    /// <summary>
    /// One guided denoiser evaluation: conditional, then unconditional, combined on the host. Two-expert models run the second
    /// (low-noise) expert with its own guidance scale below the boundary timestep; the first expert is released at the switch when the
    /// model holds one stage at a time.
    /// </summary>
    private float[] Denoise(VideoRun run, PromptRow[] conditioning, DiffusionSampler sampler, int index, float[] latents, float[]? firstFrame)
    {
        float timestep = sampler.Timesteps[index];
        bool second = _boundaryTimestep is { } boundary && timestep < boundary;
        if (second && _holdOneStage)
        {
            Release(CompiledKey(VideoDiffusionGraphs.Denoise, run.Dimensions));
        }

        var transformer = second
            ? Compiled(VideoDiffusionGraphs.SecondTransformerRole, VideoDiffusionGraphs.Denoise2, run.Dimensions)
            : Compiled(ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, run.Dimensions);
        float guidance = second ? run.Guidance2 : run.Guidance;
        var input = latents;
        if (firstFrame is not null)
        {
            input = (float[])latents.Clone();
            ReplaceFirstFrame(input, firstFrame, run);
        }

        var timesteps = TimestepInput(run, timestep, firstFrame is not null);
        var conditional = RunTransformer(transformer, run, input, timesteps, conditioning[0]);
        if (!run.Guided)
        {
            return conditional;
        }

        var unconditional = RunTransformer(transformer, run, input, timesteps, conditioning[1]);
        for (int i = 0; i < conditional.Length; i++)
        {
            conditional[i] = unconditional[i] + guidance * (conditional[i] - unconditional[i]);
        }

        return conditional;
    }

    /// <summary>The timestep input: one value, or one per token with the tokens of a conditioning first frame at 0.</summary>
    private float[] TimestepInput(VideoRun run, float timestep, bool firstFrame)
    {
        if (!_tokenTimesteps)
        {
            return [timestep];
        }

        var values = new float[TokenCount(run)];
        values.AsSpan().Fill(timestep);
        if (firstFrame)
        {
            values.AsSpan(0, run.LatentHeight / _patchSize * (run.LatentWidth / _patchSize)).Clear();
        }

        return values;
    }

    private static float[] RunTransformer(CompiledModel transformer, VideoRun run, float[] latents, float[] timesteps, PromptRow context)
    {
        transformer.SetInput<float>(Input(transformer, VideoDiffusionGraphs.Latent), latents);
        transformer.SetInput<float>(Input(transformer, VideoDiffusionGraphs.Timestep), timesteps);
        transformer.SetInput<float>(Input(transformer, VideoDiffusionGraphs.Context), context.Hidden);
        if (transformer.Graph.Inputs.Any(i => i.Name == VideoDiffusionGraphs.ContextMask))
        {
            transformer.SetInput<float>(Input(transformer, VideoDiffusionGraphs.ContextMask), context.Mask);
        }
        transformer.Run(new RunParameters());
        var prediction = new float[run.LatentSize];
        transformer.GetOutput<float>(VideoDiffusionGraphs.Prediction, prediction);
        return prediction;
    }

    /// <summary>Writes the encoded start image [C, h, w] into latent frame 0 of a latent video [C, T, h, w].</summary>
    private static void ReplaceFirstFrame(float[] latents, float[] firstFrame, VideoRun run)
    {
        int plane = run.PlaneSize;
        for (int c = 0; c < run.LatentChannels; c++)
        {
            firstFrame.AsSpan(c * plane, plane).CopyTo(latents.AsSpan(c * run.LatentFrames * plane, plane));
        }
    }

    /// <summary>Latent frame <paramref name="index"/> [C, 1, h, w] of a latent video [C, T, h, w].</summary>
    internal static float[] LatentFrame(float[] latents, int index, VideoRun run)
    {
        int plane = run.PlaneSize;
        var frame = new float[run.LatentChannels * plane];
        for (int c = 0; c < run.LatentChannels; c++)
        {
            latents.AsSpan((c * run.LatentFrames + index) * plane, plane).CopyTo(frame.AsSpan(c * plane));
        }

        return frame;
    }

    /// <summary>Decodes the whole latent video with a non-causal decoder (<see cref="VideoDiffusionGraphs.VaeDecode"/>).</summary>
    private List<ImageInput> DecodeWhole(VideoRun run, float[] latents)
    {
        var decoder = Compiled(ModelComponentRoles.Vae, VideoDiffusionGraphs.VaeDecode, run.Dimensions);
        decoder.SetInput<float>(Input(decoder, VideoDiffusionGraphs.Latent), latents);
        var parameters = new RunParameters();
        decoder.Run(parameters);
        int count = (int)decoder.OutputShape(VideoDiffusionGraphs.Frames, parameters)[0];
        int frameValues = 3 * run.Width * run.Height;
        var pixels = new float[count * frameValues];
        decoder.GetOutput<float>(VideoDiffusionGraphs.Frames, pixels);
        var frames = new List<ImageInput>(count);
        for (int f = 0; f < count; f++)
        {
            frames.Add(PlanarImage.FromPlanar(pixels.AsSpan(f * frameValues, frameValues), run.Width, run.Height, 0f, 1f));
        }

        return frames;
    }

    private CompiledModel Decoder(VideoRun run, int latentFrame)
        => Compiled(ModelComponentRoles.Vae, latentFrame == 0 ? VideoDiffusionGraphs.VaeDecodeFirst : VideoDiffusionGraphs.VaeDecodeNext, run.Dimensions);

    /// <summary>The normalized first latent frame [C, h, w] of the start image (resized to the output size).</summary>
    private float[] EncodeImage(ImageInput image, VideoRun run)
    {
        var planes = new float[3 * image.Width * image.Height];
        PlanarImage.ToPlanar(image, planes, 0f, 1f);
        var resized = image.Width == run.Width && image.Height == run.Height
            ? planes
            : PlanarResize.Resize(planes, 3, image.Height, image.Width, run.Height, run.Width);
        var encoder = Compiled(ModelComponentRoles.Vae, VideoDiffusionGraphs.VaeEncode, run.Dimensions);
        encoder.SetInput<float>(Input(encoder, VideoDiffusionGraphs.Image), resized);
        encoder.Run(new RunParameters());
        var latent = new float[run.LatentChannels * run.PlaneSize];
        encoder.GetOutput<float>(VideoDiffusionGraphs.Latent, latent);
        Release(CompiledKey(VideoDiffusionGraphs.VaeEncode, run.Dimensions));
        return latent;
    }

    /// <summary>One preview frame per latent frame (the latent RGB map at latent resolution).</summary>
    private List<ImageInput> Preview(float[] latents, VideoRun run, float[]? firstFrame)
    {
        var previews = new List<ImageInput>(run.LatentFrames);
        for (int t = 0; t < run.LatentFrames; t++)
        {
            var frame = t == 0 && firstFrame is not null ? firstFrame : LatentFrame(latents, t, run);
            previews.Add(LatentPreview.Render(frame, run.LatentChannels, run.LatentHeight, run.LatentWidth, _rgbFactors));
        }

        return previews;
    }

    private static string CompiledKey(string graph, IReadOnlyDictionary<string, long>? dimensions)
        => dimensions is null
            ? graph
            : $"{graph}@{dimensions[VideoDiffusionGraphs.Width]}x{dimensions[VideoDiffusionGraphs.Height]}x{dimensions[VideoDiffusionGraphs.FramesDimension]}";

    private CompiledModel Compiled(string role, string graph, IReadOnlyDictionary<string, long>? dimensions)
    {
        string key = CompiledKey(graph, dimensions);
        if (!_compiled.TryGetValue(key, out var compiled))
        {
            compiled = _model.CompileComponent(role, graph, null, dimensions);
            _compiled[key] = compiled;
        }

        return compiled;
    }

    private void Release(string key)
    {
        if (_compiled.Remove(key, out var compiled))
        {
            compiled.Dispose();
        }
    }

    private static Graph.Value Input(CompiledModel model, string name) => model.Graph.Inputs.Single(i => i.Name == name);
}
