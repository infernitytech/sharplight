using System.Runtime.CompilerServices;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration;

/// <summary>
/// The built-in latent-diffusion pipeline: runs any image-generation model whose description follows <see cref="DiffusionGraphs"/>
/// (text encoder(s), denoiser, VAE, tokenizer(s) and scheduler components; UNet or transformer denoisers, variance-preserving or
/// rectified-flow schedules). The prompt and the negative prompt are encoded once; per image, seeded noise (or
/// <see cref="ImageGenerationOptions.InitialNoise"/>) is denoised by the chosen sampler with classifier-free guidance on the host
/// (unconditional and conditional rows in one batched denoiser run, or one run per row for denoisers built for one row) or, for
/// guidance-distilled denoisers, with the guidance scale as a denoiser input, then decoded by the VAE. Image-to-image starts from
/// the encoded input noised to the strength's step; inpainting additionally re-imposes the noised input outside the mask after every
/// step. Graphs are compiled on first use, per image size (other sizes than the native one rebuild the model through
/// <see cref="TaskModel.DefinitionFor"/>).
/// </summary>
internal sealed class LatentDiffusionGenerator : IImageGenerator
{
    private readonly TaskModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly ITokenizer? _secondTokenizer;
    private readonly NoiseSchedule _schedule;
    private readonly int _contextLength;
    private readonly int _secondContextLength;
    private readonly string? _secondEncoderRole;
    private readonly bool _embeddedGuidance;
    private readonly bool _classifierFree;
    private readonly bool _releaseTextEncoders;
    private (string Prompt, string? Negative, bool Guided, PromptConditioning Conditioning)? _lastConditioning;
    private readonly int _patchSize;
    private readonly int _latentChannels;
    private readonly int _downscale;
    private readonly int _nativeWidth;
    private readonly int _nativeHeight;
    private readonly int _defaultSteps;
    private readonly float _defaultGuidance;
    private readonly float[] _rgbFactors;
    private readonly int _secondPadId;
    private readonly bool _zeroEmptyNegative;
    private readonly Dictionary<string, CompiledModel> _compiled = new(StringComparer.Ordinal);
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    /// <summary>Creates the pipeline for a model (the model stays owned by the caller).</summary>
    /// <param name="model">An image-generation model following <see cref="DiffusionGraphs"/>.</param>
    public LatentDiffusionGenerator(TaskModel model)
    {
        _model = model;
        var description = model.Description;
        _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        _schedule = NoiseSchedule.FromComponent(description.Component(ModelComponentRoles.Scheduler));
        var textEncoder = description.Component(ModelComponentRoles.TextEncoder);
        _contextLength = textEncoder.ConfigValue(DiffusionConfigKeys.ContextLength, 77);
        _secondEncoderRole = textEncoder.Graphs.Contains(DiffusionGraphs.TextEncode2) ? ModelComponentRoles.TextEncoder
            : description.TryGetComponent(DiffusionGraphs.SecondTextEncoderRole, out var second) && second.Graphs.Contains(DiffusionGraphs.TextEncode2)
                ? DiffusionGraphs.SecondTextEncoderRole
                : null;
        if (description.TryGetComponent(DiffusionGraphs.SecondTokenizerRole, out var secondTokenizer))
        {
            _secondTokenizer = model.CreateTokenizer(DiffusionGraphs.SecondTokenizerRole);
            _secondContextLength = secondTokenizer.ConfigValue(DiffusionConfigKeys.ContextLength, 512);
        }

        var denoiser = description.Component(ModelComponentRoles.Denoiser);
        _patchSize = denoiser.ConfigValue(DiffusionConfigKeys.PatchSize, 1);
        _embeddedGuidance = model.ComponentGraph(ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise).Inputs.Any(i => i.Name == DiffusionGraphs.Guidance);
        if (_secondEncoderRole is null && model.Definition.Graphs.TryGetValue(DiffusionGraphs.Denoise, out var denoise)
            && denoise.Inputs.Any(input => input.Name == DiffusionGraphs.Pooled)
            && !model.ComponentGraph(ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode).Outputs.ContainsKey(DiffusionGraphs.Pooled))
        {
            // An SDXL-style denoiser without the encoder that produces its pooled embedding (e.g. a package converted from a file
            // whose second text encoder was not recognized): fail on load, naming what is missing, instead of on the first image.
            throw new NotSupportedException($"{model.Definition.Name}: the denoiser is conditioned on a pooled text embedding (SDXL), but the model "
                + "has no second text encoder (OpenCLIP bigG with its text projection) to produce it. Import the original checkpoint again with this "
                + "version (it recognizes the second encoder in stable-diffusion.cpp GGUF files), or use a checkpoint that contains both text encoders.");
        }
        _secondPadId = textEncoder.ConfigValue(DiffusionConfigKeys.SecondPadId, -1);
        _zeroEmptyNegative = textEncoder.ConfigValue(DiffusionConfigKeys.ZeroEmptyNegative, false);
        var vae = description.Component(ModelComponentRoles.Vae);
        _latentChannels = vae.ConfigValue(DiffusionConfigKeys.LatentChannels, 4);
        _downscale = vae.ConfigValue(DiffusionConfigKeys.Downscale, 8);
        _rgbFactors = vae.Config.TryGetValue(DiffusionConfigKeys.LatentRgbFactors, out var factors)
            ? [.. factors.EnumerateArray().Select(f => f.GetSingle())]
            : [];
        _nativeWidth = description.ConfigValue(DiffusionConfigKeys.Width, 512);
        _nativeHeight = description.ConfigValue(DiffusionConfigKeys.Height, 512);
        _defaultSteps = description.ConfigValue(DiffusionConfigKeys.Steps, 25);
        _defaultGuidance = description.ConfigValue(DiffusionConfigKeys.GuidanceScale, 7.5f);
        _classifierFree = description.ConfigValue(DiffusionConfigKeys.ClassifierFreeGuidance, true);
        // Models that barely fit (FLUX with bf16 weights on a 32 GB GPU: a 24 GB transformer and a 9.5 GB T5) do not keep the text
        // encoders resident during denoising: they are compiled again (from the cached kernels) for the next request.
        long weightBytes = model.Definition.Weights.Values.Sum(w => w.ByteSize);
        long deviceBytes = model.Options.Devices.Sum(slot => slot.Device.Info.TotalMemory);
        _releaseTextEncoders = deviceBytes > 0 && weightBytes > deviceBytes * 0.8;
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    /// <remarks>Also answers <see cref="ImageGenerationDefaults"/> (the model's native size, steps, guidance and sampler).</remarks>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is null && serviceType == typeof(ImageGenerationDefaults))
        {
            return new ImageGenerationDefaults(_nativeWidth, _nativeHeight, _defaultSteps, _defaultGuidance, _schedule.Sampler, _classifierFree && !_embeddedGuidance);
        }

        return serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model
            : serviceKey is null && serviceType.IsInstanceOfType(this) ? this
            : null;
    }

    /// <inheritdoc/>
    public async Task<ImageGenerationResult> GenerateAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ImageGenerationResult? result = null;
        await foreach (var update in GenerateStreamingAsync(request, options, cancellationToken).ConfigureAwait(false))
        {
            result = update.Result ?? result;
        }

        return result ?? throw new InvalidOperationException("Image generation produced no result.");
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Returns at once: the pipeline runs on a background task (<see cref="BackgroundCompute"/>), one generation at a time per client, and
    /// queues its updates for the caller without waiting for them to be consumed.
    /// </remarks>
    public IAsyncEnumerable<ImageGenerationUpdate> GenerateStreamingAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
        => BackgroundCompute.StreamAsync(token => GenerateGatedAsync(request, options, token), cancellationToken);

    private async IAsyncEnumerable<ImageGenerationUpdate> GenerateGatedAsync(ImageGenerationRequest request, ImageGenerationOptions? options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await foreach (var update in GenerateCoreAsync(request, options, cancellationToken).ConfigureAwait(false))
            {
                yield return update;
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    private async IAsyncEnumerable<ImageGenerationUpdate> GenerateCoreAsync(ImageGenerationRequest request, ImageGenerationOptions? options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var run = Plan(request, options ?? ImageGenerationOptions.Default);
        var context = await Task.Run(() => Conditioning(request, run.Guided), cancellationToken).ConfigureAwait(false);
        var reference = request.Image is null ? null : await Task.Run(() => EncodeImage(request.Image, run), cancellationToken).ConfigureAwait(false);
        var mask = request.Mask is null ? null : LatentMask(request.Mask, run);
        var images = new List<ImageInput>(run.Count);
        int evaluations = CreateSampler(run).Timesteps.Count;
        int start = reference is null ? 0 : CreateSampler(run).StartIndex(run.Steps, request.Strength);
        int perImage = evaluations - start;
        int total = perImage * run.Count;
        for (int image = 0; image < run.Count; image++)
        {
            long seed = run.Seed + image;
            var sampler = CreateSampler(run);
            var noise = InitialNoise(run, image, seed);
            var latents = new float[run.LatentSize];
            if (reference is null)
            {
                for (int i = 0; i < latents.Length; i++)
                {
                    latents[i] = noise[i] * sampler.InitialNoiseScale;
                }
            }
            else if (start >= evaluations)
            {
                reference.CopyTo(latents, 0);
            }
            else
            {
                sampler.AddNoise(reference, noise, start, latents);
            }

            long stream = 1;
            for (int index = start; index < evaluations; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int step = index;
                var prediction = await Task.Run(() => Denoise(run, context, sampler, step, latents), cancellationToken).ConfigureAwait(false);
                long imageSeed = seed;
                long noiseStream = stream++;
                sampler.Step(index, prediction, latents, () => LatentNoise.Draw(imageSeed, noiseStream, latents.Length));
                if (mask is not null && reference is not null)
                {
                    BlendOutsideMask(sampler, reference, noise, mask, index, evaluations, latents);
                }

                int done = image * perImage + index - start + 1;
                bool preview = run.PreviewInterval > 0 && _rgbFactors.Length > 0 && (index - start + 1) % run.PreviewInterval == 0;
                yield return new ImageGenerationUpdate(done, total, Preview: preview ? Preview(sampler.PredictedOriginal ?? latents, run) : null);
            }

            images.Add(await Task.Run(() => Decode(run, latents), cancellationToken).ConfigureAwait(false));
        }

        yield return new ImageGenerationUpdate(total, total, new ImageGenerationResult(images, run.Seed));
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
        _gate.Dispose();
    }

    private DiffusionRun Plan(ImageGenerationRequest request, ImageGenerationOptions options)
    {
        int width = options.Width ?? request.Image?.Width ?? _nativeWidth;
        int height = options.Height ?? request.Image?.Height ?? _nativeHeight;
        if (options.Width is null && options.Height is null && request.Image is not null)
        {
            // An input image of any size is generated at the nearest size the network accepts.
            int multiple = _downscale * 8;
            width = Math.Max(multiple, width / multiple * multiple);
            height = Math.Max(multiple, height / multiple * multiple);
        }

        int steps = options.Steps ?? _defaultSteps;
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1, nameof(options));
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Count, 1, nameof(options));
        if (request.Mask is not null && request.Image is null)
        {
            throw new ArgumentException("Inpainting needs ImageGenerationRequest.Image as well as the mask.", nameof(request));
        }

        float guidance = options.GuidanceScale ?? _defaultGuidance;
        var run = new DiffusionRun
        {
            Width = width,
            Height = height,
            LatentWidth = width / _downscale,
            LatentHeight = height / _downscale,
            LatentChannels = _latentChannels,
            Steps = steps,
            Guidance = guidance,
            Guided = guidance > 1f && _classifierFree && !_embeddedGuidance,
            Seed = options.Seed ?? Random.Shared.NextInt64(1L << 53),
            Count = options.Count,
            Sampler = options.Scheduler ?? _schedule.Sampler,
            PreviewInterval = options.PreviewInterval,
            InitialNoise = options.InitialNoise,
            Dimensions = width == _nativeWidth && height == _nativeHeight
                ? null
                : new Dictionary<string, long> { [DiffusionGraphs.Height] = height, [DiffusionGraphs.Width] = width },
        };
        if (run.InitialNoise is not null && run.InitialNoise.Length < run.LatentSize * run.Count)
        {
            throw new ArgumentException($"InitialNoise has {run.InitialNoise.Length} values; {run.Count} image(s) of {run.LatentSize} need {run.LatentSize * run.Count}.", nameof(options));
        }

        return run;
    }

    /// <summary>
    /// The conditioning of a request, reused when the prompts are those of the previous request. Models that barely fit the device
    /// hold either the text encoders or the denoiser: the other components are released before encoding and the encoders after it.
    /// </summary>
    private PromptConditioning Conditioning(ImageGenerationRequest request, bool guided)
    {
        if (_lastConditioning is { } last && last.Prompt == request.Prompt && last.Negative == request.NegativePrompt && last.Guided == guided)
        {
            return last.Conditioning;
        }

        if (_releaseTextEncoders)
        {
            foreach (string key in _compiled.Keys.Where(k => k != DiffusionGraphs.TextEncode && k != DiffusionGraphs.TextEncode2).ToList())
            {
                Release(key);
            }
        }

        var conditioning = EncodePrompts(request, guided);
        if (_releaseTextEncoders)
        {
            Release(DiffusionGraphs.TextEncode);
            Release(DiffusionGraphs.TextEncode2);
        }

        _lastConditioning = (request.Prompt, request.NegativePrompt, guided, conditioning);
        return conditioning;
    }

    /// <summary>The text conditioning: the negative prompt's row first when guided, then the prompt's.</summary>
    private PromptConditioning EncodePrompts(ImageGenerationRequest request, bool guided)
    {
        var conditional = EncodePrompt(request.Prompt);
        if (!guided)
        {
            return conditional;
        }

        var unconditional = request.NegativePrompt is null && _zeroEmptyNegative ? conditional.Zeros() : EncodePrompt(request.NegativePrompt ?? "");
        return unconditional.Append(conditional);
    }

    /// <summary>
    /// One prompt's conditioning: the hidden states of the text encoders joined per token (SDXL: both CLIP encoders; FLUX: only the
    /// T5 encoder has them) and the pooled embedding of the second encoder, else of the first (SDXL's OpenCLIP, FLUX's CLIP).
    /// </summary>
    private PromptConditioning EncodePrompt(string prompt)
    {
        var tokens = PromptTokens(_tokenizer, _contextLength, prompt);
        var (hidden, pooled) = RunTextEncoder(ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, tokens);
        if (_secondEncoderRole is null)
        {
            return new PromptConditioning(hidden ?? throw new InvalidOperationException("The text encoder produces no hidden states."), pooled);
        }

        int[] secondTokens;
        if (_secondTokenizer is not null)
        {
            secondTokens = PromptTokens(_secondTokenizer, _secondContextLength, prompt);
        }
        else
        {
            secondTokens = (int[])tokens.Clone();
            int end = Array.IndexOf(secondTokens, _tokenizer.EosId ?? secondTokens[^1]);
            if (_secondPadId >= 0 && end >= 0)
            {
                secondTokens.AsSpan(end + 1).Fill(_secondPadId);
            }
        }

        var (secondHidden, secondPooled) = RunTextEncoder(_secondEncoderRole, DiffusionGraphs.TextEncode2, secondTokens);
        return new PromptConditioning(JoinPerToken(hidden, tokens.Length, secondHidden, secondTokens.Length), secondPooled ?? pooled);
    }

    /// <summary>The per-token concatenation of two encoders' hidden states (either may be absent).</summary>
    private static float[] JoinPerToken(float[]? first, int firstRows, float[]? second, int secondRows)
    {
        if (first is null || second is null)
        {
            return first ?? second ?? throw new InvalidOperationException("No text encoder produces hidden states.");
        }

        if (firstRows != secondRows)
        {
            throw new NotSupportedException($"The text encoders' hidden states have different lengths ({firstRows}, {secondRows}).");
        }

        int rows = firstRows;
        int a = first.Length / rows;
        int b = second.Length / rows;
        var context = new float[rows * (a + b)];
        for (int row = 0; row < rows; row++)
        {
            first.AsSpan(row * a, a).CopyTo(context.AsSpan(row * (a + b)));
            second.AsSpan(row * b, b).CopyTo(context.AsSpan(row * (a + b) + a));
        }

        return context;
    }

    private (float[]? Hidden, float[]? Pooled) RunTextEncoder(string role, string graph, int[] tokens)
    {
        var encoder = Compiled(role, graph, null);
        encoder.SetInput<int>(encoder.Graph.Inputs.Single(i => i.Name == DiffusionGraphs.Tokens), tokens);
        var parameters = new RunParameters();
        encoder.Run(parameters);
        return (Output(encoder, DiffusionGraphs.Hidden, parameters), Output(encoder, DiffusionGraphs.Pooled, parameters));
    }

    private static float[]? Output(CompiledModel model, string name, RunParameters parameters)
    {
        if (!model.Graph.Outputs.ContainsKey(name))
        {
            return null;
        }

        var values = new float[model.OutputShape(name, parameters).Aggregate(1L, (a, b) => a * b)];
        model.GetOutput<float>(name, values);
        return values;
    }

    /// <summary>
    /// Tokenizes a prompt to exactly the context length: the tokenizer's special tokens (CLIP: start and end; T5: end), truncated to
    /// keep the end token, then padding.
    /// </summary>
    private static int[] PromptTokens(ITokenizer tokenizer, int contextLength, string prompt)
    {
        var ids = tokenizer.Encode(prompt, addSpecial: true, parseSpecial: false).ToList();
        int end = tokenizer.EosId ?? ids[^1];
        if (ids.Count > contextLength)
        {
            ids = [.. ids.Take(contextLength - 1), end];
        }

        int pad = tokenizer.PadId ?? end;
        while (ids.Count < contextLength)
        {
            ids.Add(pad);
        }

        return [.. ids];
    }

    private float[] Denoise(DiffusionRun run, PromptConditioning conditioning, DiffusionSampler sampler, int index, float[] latents)
    {
        var denoiser = Compiled(ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, run.Dimensions);
        int rows = run.Guided ? 2 : 1;
        var batchDim = Input(denoiser, DiffusionGraphs.Latent).Shape[0];
        int perRun = batchDim.IsSymbolic ? rows : (int)batchDim.Fixed;
        var output = new float[run.LatentSize * rows];
        if (perRun >= rows)
        {
            RunDenoiser(denoiser, run, conditioning, sampler, index, latents, 0, rows, output);
        }
        else
        {
            // A denoiser built for fewer rows (a fixed batch of one) runs once per row of a guided step.
            for (int row = 0; row < rows; row++)
            {
                RunDenoiser(denoiser, run, conditioning, sampler, index, latents, row, 1, output.AsSpan(row * run.LatentSize, run.LatentSize));
            }
        }

        if (!run.Guided)
        {
            return output;
        }

        var guided = new float[run.LatentSize];
        for (int i = 0; i < guided.Length; i++)
        {
            float unconditional = output[i];
            float conditional = output[run.LatentSize + i];
            guided[i] = unconditional + run.Guidance * (conditional - unconditional);
        }

        return guided;
    }

    /// <summary>One denoiser run over <paramref name="rows"/> conditioning rows starting at <paramref name="firstRow"/>.</summary>
    private static void RunDenoiser(CompiledModel denoiser, DiffusionRun run, PromptConditioning conditioning, DiffusionSampler sampler, int index,
        float[] latents, int firstRow, int rows, Span<float> output)
    {
        float scale = sampler.ModelInputScale(index);
        var input = new float[run.LatentSize * rows];
        for (int row = 0; row < rows; row++)
        {
            for (int i = 0; i < latents.Length; i++)
            {
                input[row * run.LatentSize + i] = latents[i] * scale;
            }
        }

        int totalRows = run.Guided ? 2 : 1;
        int contextRow = conditioning.Context.Length / totalRows;
        denoiser.SetInput<float>(Input(denoiser, DiffusionGraphs.Latent), input);
        denoiser.SetInput<float>(Input(denoiser, DiffusionGraphs.Timestep), Enumerable.Repeat(sampler.Timesteps[index], rows).ToArray());
        denoiser.SetInput<float>(Input(denoiser, DiffusionGraphs.Context), conditioning.Context.AsSpan(firstRow * contextRow, rows * contextRow).ToArray());
        if (denoiser.Graph.Inputs.Any(i => i.Name == DiffusionGraphs.Pooled))
        {
            var pooled = conditioning.Pooled ?? throw new InvalidOperationException("The denoiser needs a pooled text embedding, which no text encoder produced.");
            int pooledRow = pooled.Length / totalRows;
            denoiser.SetInput<float>(Input(denoiser, DiffusionGraphs.Pooled), pooled.AsSpan(firstRow * pooledRow, rows * pooledRow).ToArray());
        }

        if (denoiser.Graph.Inputs.Any(i => i.Name == DiffusionGraphs.SizeConditions))
        {
            // Original size = target size = the output size, no crop (diffusers' defaults).
            float[] size = [run.Height, run.Width, 0f, 0f, run.Height, run.Width];
            var sizes = new float[size.Length * rows];
            for (int row = 0; row < rows; row++)
            {
                size.CopyTo(sizes, row * size.Length);
            }

            denoiser.SetInput<float>(Input(denoiser, DiffusionGraphs.SizeConditions), sizes);
        }

        if (denoiser.Graph.Inputs.Any(i => i.Name == DiffusionGraphs.Guidance))
        {
            denoiser.SetInput<float>(Input(denoiser, DiffusionGraphs.Guidance), Enumerable.Repeat(run.Guidance, rows).ToArray());
        }

        var parameters = new RunParameters();
        parameters.Symbols[DiffusionGraphs.BatchSymbol] = rows;
        denoiser.Run(parameters);
        var result = new float[run.LatentSize * rows];
        denoiser.GetOutput<float>(DiffusionGraphs.Prediction, result);
        result.CopyTo(output);
    }

    private ImageInput Decode(DiffusionRun run, float[] latents)
    {
        var decoder = Compiled(ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, run.Dimensions);
        decoder.SetInput<float>(Input(decoder, DiffusionGraphs.Latent), latents);
        var parameters = new RunParameters();
        parameters.Symbols[DiffusionGraphs.BatchSymbol] = 1;
        decoder.Run(parameters);
        var pixels = new float[3 * run.Width * run.Height];
        decoder.GetOutput<float>(DiffusionGraphs.Image, pixels);
        return PlanarImage.FromPlanar(pixels, run.Width, run.Height, 0f, 1f);
    }

    /// <summary>The model latents of the input image (resized to the output size): the VAE encoder's scaled mean.</summary>
    private float[] EncodeImage(ImageInput image, DiffusionRun run)
    {
        var planes = new float[3 * image.Width * image.Height];
        PlanarImage.ToPlanar(image, planes, 0f, 1f);
        var resized = PlanarResize.Resize(planes, 3, image.Height, image.Width, run.Height, run.Width);
        var encoder = Compiled(ModelComponentRoles.Vae, DiffusionGraphs.VaeEncode, run.Dimensions);
        encoder.SetInput<float>(Input(encoder, DiffusionGraphs.Image), resized);
        var parameters = new RunParameters();
        parameters.Symbols[DiffusionGraphs.BatchSymbol] = 1;
        encoder.Run(parameters);
        var latents = new float[run.LatentSize];
        encoder.GetOutput<float>(DiffusionGraphs.Latent, latents);
        return latents;
    }

    /// <summary>The mask at latent resolution (nearest sampling, 1 = regenerate).</summary>
    private static float[] LatentMask(ImageInput mask, DiffusionRun run)
    {
        var planes = new float[3 * mask.Width * mask.Height];
        PlanarImage.ToPlanar(mask, planes, 0f, 1f);
        var resized = PlanarResize.Resize(planes, 3, mask.Height, mask.Width, run.Height, run.Width);
        int pixels = run.Width * run.Height;
        var latentMask = new float[run.LatentWidth * run.LatentHeight];
        for (int y = 0; y < run.LatentHeight; y++)
        {
            for (int x = 0; x < run.LatentWidth; x++)
            {
                int source = y * run.Height / run.LatentHeight * run.Width + x * run.Width / run.LatentWidth;
                float intensity = (resized[source] + resized[pixels + source] + resized[2 * pixels + source]) / 3f;
                latentMask[y * run.LatentWidth + x] = intensity >= 0.5f ? 1f : 0f;
            }
        }

        return latentMask;
    }

    /// <summary>Inpainting: outside the mask the sample becomes the input image noised to the next step's level (clean after the last).</summary>
    private static void BlendOutsideMask(DiffusionSampler sampler, float[] reference, float[] noise, float[] mask, int index, int evaluations,
        float[] latents)
    {
        var known = reference;
        if (index < evaluations - 1)
        {
            known = new float[latents.Length];
            sampler.AddNoise(reference, noise, index + 1, known);
        }

        int plane = mask.Length;
        for (int i = 0; i < latents.Length; i++)
        {
            float m = mask[i % plane];
            latents[i] = (1 - m) * known[i] + m * latents[i];
        }
    }

    /// <summary>A sampler for one image of the run (flow samplers get the denoiser's image token count for the dynamic shift).</summary>
    private DiffusionSampler CreateSampler(DiffusionRun run)
        => Samplers.Create(run.Sampler, _schedule, run.Steps, run.LatentWidth / _patchSize * (run.LatentHeight / _patchSize));

    private float[] InitialNoise(DiffusionRun run, int image, long seed)
        => run.InitialNoise is { } explicitNoise
            ? explicitNoise.AsSpan(image * run.LatentSize, run.LatentSize).ToArray()
            : LatentNoise.Draw(seed, 0, run.LatentSize);

    private ImageInput Preview(float[] latents, DiffusionRun run)
        => LatentPreview.Render(latents, run.LatentChannels, run.LatentHeight, run.LatentWidth, _rgbFactors);

    private CompiledModel Compiled(string role, string graph, IReadOnlyDictionary<string, long>? dimensions)
    {
        string key = dimensions is null ? graph : $"{graph}@{dimensions[DiffusionGraphs.Width]}x{dimensions[DiffusionGraphs.Height]}";
        if (!_compiled.TryGetValue(key, out var compiled))
        {
            compiled = _model.CompileComponent(role, graph, null, dimensions);
            _compiled[key] = compiled;
        }

        return compiled;
    }

    private void Release(string graph)
    {
        if (_compiled.Remove(graph, out var compiled))
        {
            compiled.Dispose();
        }
    }

    private static Graph.Value Input(CompiledModel model, string name) => model.Graph.Inputs.Single(i => i.Name == name);
}
