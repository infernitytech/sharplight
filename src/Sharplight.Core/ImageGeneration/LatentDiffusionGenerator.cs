using System.Runtime.CompilerServices;
using Sharplight.Architectures;
using Sharplight.Imaging;
using Sharplight.Llm;
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
    private readonly ITokenizer? _thirdTokenizer;
    private readonly int _thirdContextLength;
    private readonly bool _thirdJoinsSequence;
    private readonly int _contextMaskPadding = -1;
    private readonly int _promptSkip;
    private readonly bool _guidanceRenormalization;
    private readonly string? _promptTemplate;
    private readonly string? _emptyPrompt;
    private readonly bool _embeddedGuidance;
    private readonly int _maxReferences;
    private readonly string? _conditionMode;
    private readonly VisionParameters? _vision;
    private readonly long _conditionPixels;
    private readonly long _referenceArea;
    private readonly string _imagePrompt = "";
    private readonly float _defaultImageGuidance;
    private readonly float _scalingFactor;
    private readonly bool _classifierFree;
    private bool _releaseTextEncoders;
    private (string Prompt, string? Negative, int Rows, IReadOnlyList<ImageInput>? Images, PromptConditioning Conditioning)? _lastConditioning;
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
        _promptTemplate = description.Component(ModelComponentRoles.Tokenizer).ConfigValue<string?>(DiffusionConfigKeys.PromptTemplate, null);
        _emptyPrompt = description.Component(ModelComponentRoles.Tokenizer).ConfigValue<string?>(DiffusionConfigKeys.EmptyPrompt, null);
        var textEncoder = description.Component(ModelComponentRoles.TextEncoder);
        _contextLength = textEncoder.ConfigValue(DiffusionConfigKeys.ContextLength, 77);
        _promptSkip = textEncoder.ConfigValue(DiffusionConfigKeys.PromptSkip, 0);
        _conditionPixels = textEncoder.ConfigValue(DiffusionConfigKeys.ConditionPixels, 0L);
        if (_conditionPixels > 0 && textEncoder.Graphs.Contains(WellKnown.VisionEncodeGraph))
        {
            // Qwen-Image-Edit: the language model reads the reference images through its vision tower.
            _vision = VisionParameters.Of(model.Definition);
            _imagePrompt = description.Component(ModelComponentRoles.Tokenizer).ConfigValue(DiffusionConfigKeys.ImagePrompt, "");
        }

        _referenceArea = description.ConfigValue(DiffusionConfigKeys.ReferenceArea, 0L);
        _secondEncoderRole = textEncoder.Graphs.Contains(DiffusionGraphs.TextEncode2) ? ModelComponentRoles.TextEncoder
            : description.TryGetComponent(DiffusionGraphs.SecondTextEncoderRole, out var second) && second.Graphs.Contains(DiffusionGraphs.TextEncode2)
                ? DiffusionGraphs.SecondTextEncoderRole
                : null;
        if (description.TryGetComponent(DiffusionGraphs.SecondTokenizerRole, out var secondTokenizer))
        {
            _secondTokenizer = model.CreateTokenizer(DiffusionGraphs.SecondTokenizerRole);
            _secondContextLength = secondTokenizer.ConfigValue(DiffusionConfigKeys.ContextLength, 512);
        }

        if (description.TryGetComponent(DiffusionGraphs.ThirdTextEncoderRole, out var third) && third.Graphs.Contains(DiffusionGraphs.TextEncode3))
        {
            _thirdTokenizer = model.CreateTokenizer(description.TryGetComponent(DiffusionGraphs.ThirdTokenizerRole, out _)
                ? DiffusionGraphs.ThirdTokenizerRole
                : ModelComponentRoles.Tokenizer);
            _thirdContextLength = description.TryGetComponent(DiffusionGraphs.ThirdTokenizerRole, out var thirdTokenizer)
                ? thirdTokenizer.ConfigValue(DiffusionConfigKeys.ContextLength, 256)
                : third.ConfigValue(DiffusionConfigKeys.ContextLength, 256);
            _thirdJoinsSequence = third.ConfigValue(DiffusionConfigKeys.JoinSequence, false);
        }

        var denoiser = description.Component(ModelComponentRoles.Denoiser);
        _patchSize = denoiser.ConfigValue(DiffusionConfigKeys.PatchSize, 1);
        _guidanceRenormalization = denoiser.ConfigValue(DiffusionConfigKeys.GuidanceRenormalization, false);
        if (model.ComponentGraph(ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise).Inputs.Any(i => i.Name == DiffusionGraphs.ContextMask))
        {
            _contextMaskPadding = denoiser.ConfigValue(DiffusionConfigKeys.ContextMaskPadding, 0);
        }

        _maxReferences = denoiser.ConfigValue(DiffusionConfigKeys.ReferenceImages, 0);
        _conditionMode = model.ComponentGraph(ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise).Inputs.Any(i => i.Name == DiffusionGraphs.ConditionLatent)
            ? denoiser.ConfigValue<string?>(DiffusionConfigKeys.ConditionMode, null)
            : null;
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
        _scalingFactor = vae.ConfigValue(DiffusionConfigKeys.ScalingFactor, 1f);
        _downscale = vae.ConfigValue(DiffusionConfigKeys.Downscale, 8);
        _rgbFactors = vae.Config.TryGetValue(DiffusionConfigKeys.LatentRgbFactors, out var factors)
            ? [.. factors.EnumerateArray().Select(f => f.GetSingle())]
            : [];
        _nativeWidth = description.ConfigValue(DiffusionConfigKeys.Width, 512);
        _nativeHeight = description.ConfigValue(DiffusionConfigKeys.Height, 512);
        _defaultSteps = description.ConfigValue(DiffusionConfigKeys.Steps, 25);
        _defaultGuidance = description.ConfigValue(DiffusionConfigKeys.GuidanceScale, 7.5f);
        _defaultImageGuidance = _conditionMode == DiffusionConfigKeys.ConditionImage ? description.ConfigValue(DiffusionConfigKeys.ImageGuidanceScale, 1.5f) : 0f;
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
            return new ImageGenerationDefaults(_nativeWidth, _nativeHeight, _defaultSteps, _defaultGuidance, _schedule.Sampler, _classifierFree && !_embeddedGuidance, _maxReferences, _defaultImageGuidance);
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
        var context = await Task.Run(() => Conditioning(request, run.Rows), cancellationToken).ConfigureAwait(false);
        bool imageConditioned = _conditionMode == DiffusionConfigKeys.ConditionImage;
        bool inpaintConditioned = _conditionMode == DiffusionConfigKeys.ConditionInpaint;
        // InstructPix2Pix starts from pure noise (its image enters as extra channels); an inpainting UNet at full strength does as well.
        bool startsFromImage = request.Image is not null && !imageConditioned && !(inpaintConditioned && request.Strength >= 1f);
        var reference = request.Image is null || !(startsFromImage || inpaintConditioned && request.Mask is not null)
            ? null
            : await Task.Run(() => EncodeImage(request.Image, run), cancellationToken).ConfigureAwait(false);
        var mask = request.Mask is null ? null : LatentMask(request.Mask, run);
        var condition = _conditionMode is null ? null : await Task.Run(() => ConditionLatents(request, run, mask), cancellationToken).ConfigureAwait(false);
        var references = run.References == 0 ? null : await Task.Run(() => EncodeReferences(request.ReferenceImages!, run), cancellationToken).ConfigureAwait(false);
        var images = new List<ImageInput>(run.Count);
        int evaluations = CreateSampler(run).Timesteps.Count;
        int start = !startsFromImage ? 0 : CreateSampler(run).StartIndex(run.Steps, request.Strength);
        int perImage = evaluations - start;
        int total = perImage * run.Count;
        for (int image = 0; image < run.Count; image++)
        {
            long seed = run.Seed + image;
            var sampler = CreateSampler(run);
            var noise = InitialNoise(run, image, seed);
            var latents = new float[run.LatentSize];
            if (!startsFromImage)
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
                var prediction = await Task.Run(() => Denoise(run, context, sampler, step, latents, references, condition), cancellationToken).ConfigureAwait(false);
                long imageSeed = seed;
                long noiseStream = stream++;
                sampler.Step(index, prediction, latents, () => LatentNoise.Draw(imageSeed, noiseStream, latents.Length));
                if (mask is not null && reference is not null && !inpaintConditioned)
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
        int references = request.ReferenceImages?.Count ?? 0;
        if (references > _maxReferences)
        {
            throw new NotSupportedException(_maxReferences == 0
                ? $"{_model.Definition.Name} does not edit images (no reference images); use Image for image-to-image."
                : $"{_model.Definition.Name} takes at most {_maxReferences} reference image(s); {references} were given.");
        }

        var sizeSource = request.Image ?? (references > 0 ? request.ReferenceImages![_referenceArea > 0 ? references - 1 : 0] : null);
        int width = options.Width ?? sizeSource?.Width ?? _nativeWidth;
        int height = options.Height ?? sizeSource?.Height ?? _nativeHeight;
        if (options.Width is null && options.Height is null && request.Image is null && references > 0 && _referenceArea > 0)
        {
            // Qwen-Image-Edit: about 1024 · 1024 pixels at the last reference image's aspect ratio, sides rounded to 32.
            (width, height) = FitArea(_referenceArea, sizeSource!.Width, sizeSource.Height);
        }
        else if (options.Width is null && options.Height is null && sizeSource is not null)
        {
            // An input image of any size is generated at the nearest size the network accepts; editing models (transformers) take
            // multiples of their token size, as their pipelines round reference images.
            int multiple = request.Image is null ? _downscale * _patchSize : _downscale * 8;
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
        float imageGuidance = _conditionMode == DiffusionConfigKeys.ConditionImage ? options.ImageGuidanceScale ?? _defaultImageGuidance : 0f;
        if (_conditionMode == DiffusionConfigKeys.ConditionImage && (request.Image is null || request.Mask is not null))
        {
            throw new ArgumentException($"{_model.Definition.Name} edits an image by an instruction: give ImageGenerationRequest.Image (and no mask).", nameof(request));
        }

        if (_conditionMode == DiffusionConfigKeys.ConditionInpaint && request.Image is null)
        {
            throw new ArgumentException($"{_model.Definition.Name} is an inpainting model: give ImageGenerationRequest.Image (and Mask).", nameof(request));
        }

        var run = new DiffusionRun
        {
            Width = width,
            Height = height,
            LatentWidth = width / _downscale,
            LatentHeight = height / _downscale,
            LatentChannels = _latentChannels,
            Steps = steps,
            Guidance = guidance,
            Guided = guidance > 1f && _classifierFree && !_embeddedGuidance && imageGuidance is 0f or >= 1f,
            ImageGuidance = imageGuidance,
            Seed = options.Seed ?? Random.Shared.NextInt64(1L << 53),
            Count = options.Count,
            Sampler = options.Scheduler ?? _schedule.Sampler,
            PreviewInterval = options.PreviewInterval,
            InitialNoise = options.InitialNoise,
            References = references,
            Dimensions = width == _nativeWidth && height == _nativeHeight && references == 0
                ? null
                : references == 0
                    ? new Dictionary<string, long> { [DiffusionGraphs.Height] = height, [DiffusionGraphs.Width] = width }
                    : new Dictionary<string, long> { [DiffusionGraphs.Height] = height, [DiffusionGraphs.Width] = width, [DiffusionGraphs.References] = references },
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
    private PromptConditioning Conditioning(ImageGenerationRequest request, int rows)
    {
        if (_lastConditioning is { } last && last.Prompt == request.Prompt && last.Negative == request.NegativePrompt && last.Rows == rows
            && ReferenceEquals(last.Images, request.ReferenceImages))
        {
            return last.Conditioning;
        }

        if (_releaseTextEncoders)
        {
            foreach (string key in _compiled.Keys.Where(k => !IsTextGraph(k)).ToList())
            {
                Release(key);
            }
        }

        var conditioning = EncodePrompts(request, rows);
        if (_releaseTextEncoders)
        {
            Release(DiffusionGraphs.TextEncode);
            Release(DiffusionGraphs.TextEncode2);
            Release(DiffusionGraphs.TextEncode3);
            Release(WellKnown.VisionEncodeGraph);
        }

        _lastConditioning = (request.Prompt, request.NegativePrompt, rows, request.ReferenceImages, conditioning);
        return conditioning;
    }

    /// <summary>The text conditioning: the negative prompt's row first when guided, then the prompt's.</summary>
    private PromptConditioning EncodePrompts(ImageGenerationRequest request, int rows)
    {
        var pictures = request.ReferenceImages is { Count: > 0 } references && _vision is not null ? EncodePictures(references) : null;
        var conditional = EncodePrompt(request.Prompt, pictures);
        if (rows == 1)
        {
            return conditional;
        }

        var unconditional = request.NegativePrompt is null && _zeroEmptyNegative ? conditional.Zeros() : EncodePrompt(request.NegativePrompt ?? "", pictures);
        // Image guidance (InstructPix2Pix): the instruction's row, then the negative prompt's twice (with and without the image).
        return rows == 3 ? conditional.Append(unconditional).Append(unconditional) : unconditional.Append(conditional);
    }

    /// <summary>
    /// One prompt's conditioning: the hidden states of the text encoders joined per token (SDXL: both CLIP encoders; FLUX: only the
    /// T5 encoder has them), a third encoder's appended along the sequence when it says so (SD3's T5), and the pooled embeddings of all
    /// encoders concatenated in order (SDXL: the OpenCLIP one; FLUX: the CLIP one; SD3: CLIP-L then CLIP-G).
    /// </summary>
    private PromptConditioning EncodePrompt(string prompt, ImagePrompt? pictures = null)
    {
        if (prompt.Length == 0 && _emptyPrompt is not null)
        {
            prompt = _emptyPrompt;
        }

        ImageRows? imageRows = null;
        var (tokens, length) = pictures is null ? PromptTokens(_tokenizer, _contextLength, prompt, _promptTemplate) : PictureTokens(prompt, pictures, out imageRows);
        var (hidden, pooled) = RunTextEncoder(ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, tokens, length, imageRows);
        int rows = tokens.Length;
        if (_promptSkip > 0 && hidden is not null)
        {
            // The template's leading tokens (Qwen-Image's system prompt) condition the prompt's hidden states but are not part of the context.
            int width = hidden.Length / rows;
            hidden = hidden.AsSpan(_promptSkip * width).ToArray();
            rows -= _promptSkip;
            length = Math.Max(0, length - _promptSkip);
        }

        if (_secondEncoderRole is not null)
        {
            int[] secondTokens;
            if (_secondTokenizer is not null)
            {
                (secondTokens, _) = PromptTokens(_secondTokenizer, _secondContextLength, prompt);
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

            var (secondHidden, secondPooled) = RunTextEncoder(_secondEncoderRole, DiffusionGraphs.TextEncode2, secondTokens, secondTokens.Length);
            (hidden, rows) = Join(hidden, rows, secondHidden, secondTokens.Length, sequence: false);
            pooled = Concat(pooled, secondPooled);
        }

        if (_thirdTokenizer is not null)
        {
            var (thirdTokens, thirdLength) = PromptTokens(_thirdTokenizer, _thirdContextLength, prompt);
            var (thirdHidden, thirdPooled) = RunTextEncoder(DiffusionGraphs.ThirdTextEncoderRole, DiffusionGraphs.TextEncode3, thirdTokens, thirdLength);
            (hidden, rows) = Join(hidden, rows, thirdHidden, thirdTokens.Length, _thirdJoinsSequence);
            pooled = Concat(pooled, thirdPooled);
        }

        float[]? mask = null;
        if (_contextMaskPadding >= 0)
        {
            // A masked denoiser (one per-token encoder): the prompt's tokens and the configured number of padding tokens stay visible.
            mask = new float[rows];
            mask.AsSpan(0, Math.Min(rows, length + _contextMaskPadding)).Fill(1f);
        }

        return new PromptConditioning(hidden ?? throw new InvalidOperationException("No text encoder produces hidden states."), pooled, mask);
    }

    /// <summary>
    /// The language model's view of the reference images (Qwen-Image-Edit): each image resized to <see cref="_conditionPixels"/> pixels (its
    /// aspect ratio kept, sides multiples of 32), preprocessed and run through the vision tower.
    /// </summary>
    private ImagePrompt EncodePictures(IReadOnlyList<ImageInput> images)
    {
        var encoder = VisionEncoder.FromCompiled(_model.Definition, Compiled(ModelComponentRoles.TextEncoder, WellKnown.VisionEncodeGraph, null),
            _contextLength - _promptSkip, new Qwen35Multimodal());
        var preprocessed = new List<PreprocessedImage>(images.Count);
        var embeddings = new List<float[]>(images.Count);
        foreach (var image in images)
        {
            var (width, height) = FitArea(_conditionPixels, image.Width, image.Height);
            var planes = new float[3 * image.Width * image.Height];
            PlanarImage.ToPlanar(image, planes, 0f, 1f);
            var condition = PlanarImage.FromPlanar(PlanarResize.Resize(planes, 3, image.Height, image.Width, height, width, lanczos: true), width, height, 0f, 1f);
            var pre = encoder.Preprocess(condition);
            preprocessed.Add(pre);
            embeddings.Add(encoder.Encode(pre));
        }

        return new ImagePrompt(preprocessed, embeddings);
    }

    /// <summary>
    /// The templated prompt with the image labels and placeholders before the prompt, the placeholders expanded to the images' rows (their
    /// M-RoPE positions and embeddings in <paramref name="rows"/>), padded to the encoder's rows.
    /// </summary>
    private (int[] Ids, int Length) PictureTokens(string prompt, ImagePrompt pictures, out ImageRows rows)
    {
        string template = _promptTemplate ?? "{prompt}";
        int at = template.IndexOf("{prompt}", StringComparison.Ordinal);
        var labels = new System.Text.StringBuilder();
        for (int n = 1; n <= pictures.Images.Count; n++)
        {
            labels.Append(_imagePrompt.Replace("{n}", n.ToString(System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal));
        }

        int[] ids = [.. _tokenizer.Encode(template[..at] + labels, addSpecial: false, parseSpecial: true),
            .. _tokenizer.Encode(prompt, addSpecial: false, parseSpecial: false),
            .. _tokenizer.Encode(template[(at + "{prompt}".Length)..], addSpecial: false, parseSpecial: true)];
        int? Lookup(string text)
        {
            var token = _tokenizer.Encode(text, addSpecial: false, parseSpecial: true);
            return token.Length == 1 ? token[0] : null;
        }

        var assembled = new Qwen35Multimodal().Assemble(new MultimodalContext(_model.Definition, Lookup), ids, pictures.Images, 0);
        int total = _contextLength;
        if (assembled.Length > total)
        {
            throw new NotSupportedException($"The prompt with its {pictures.Images.Count} image(s) takes {assembled.Length} rows, more than the text encoder's {total} "
                + $"(the template's {_promptSkip} and {total - _promptSkip} for the prompt and the images): use fewer or smaller images or a shorter prompt.");
        }

        int axes = assembled.PositionAxes;
        int width = _vision!.EmbeddingWidth;
        var tokens = new int[total];
        var positions = new int[total * axes];
        var embeddings = new float[(long)total * width];
        var mask = new float[total];
        int pad = _tokenizer.PadId ?? _tokenizer.EosId ?? 0;
        for (int row = 0; row < total; row++)
        {
            if (row >= assembled.Length)
            {
                tokens[row] = pad;
                positions.AsSpan(row * axes, Math.Min(axes, 3)).Fill(assembled.NextPosition + row - assembled.Length);
                continue;
            }

            tokens[row] = assembled.Tokens[row];
            assembled.Positions.AsSpan(row * axes, axes).CopyTo(positions.AsSpan(row * axes));
            if (assembled.ImageIndex[row] >= 0)
            {
                pictures.Embeddings[assembled.ImageIndex[row]].AsSpan(assembled.ImageRow[row] * width, width).CopyTo(embeddings.AsSpan(row * width));
                mask[row] = 1f;
            }
        }

        rows = new ImageRows(positions, embeddings, mask);
        return (tokens, assembled.Length);
    }

    /// <summary>The size of the image with the given aspect ratio of about <paramref name="area"/> pixels, sides rounded to multiples of 32 (diffusers' calculate_dimensions).</summary>
    private static (int Width, int Height) FitArea(long area, int width, int height)
    {
        double ratio = (double)width / height;
        double w = Math.Sqrt(area * ratio);
        double h = w / ratio;
        return ((int)Math.Round(w / 32) * 32, (int)Math.Round(h / 32) * 32);
    }

    private static float[]? Concat(float[]? first, float[]? second)
        => first is null ? second : second is null ? first : [.. first, .. second];

    /// <summary>
    /// Joins two encoders' hidden states (either may be absent): per token ([L, D1 + D2]), or along the sequence with the narrower rows
    /// zero-padded to the wider width ([L1 + L2, max(D1, D2)]). Returns the joined states and their row count.
    /// </summary>
    private static (float[]? Hidden, int Rows) Join(float[]? first, int firstRows, float[]? second, int secondRows, bool sequence)
    {
        if (first is null || second is null)
        {
            return first is null ? (second, secondRows) : (first, firstRows);
        }

        int a = first.Length / firstRows;
        int b = second.Length / secondRows;
        if (sequence)
        {
            int width = Math.Max(a, b);
            var joined = new float[(firstRows + secondRows) * width];
            for (int row = 0; row < firstRows; row++)
            {
                first.AsSpan(row * a, a).CopyTo(joined.AsSpan(row * width));
            }

            for (int row = 0; row < secondRows; row++)
            {
                second.AsSpan(row * b, b).CopyTo(joined.AsSpan((firstRows + row) * width));
            }

            return (joined, firstRows + secondRows);
        }

        if (firstRows != secondRows)
        {
            throw new NotSupportedException($"The text encoders' hidden states have different lengths ({firstRows}, {secondRows}).");
        }

        int rows = firstRows;
        var context = new float[rows * (a + b)];
        for (int row = 0; row < rows; row++)
        {
            first.AsSpan(row * a, a).CopyTo(context.AsSpan(row * (a + b)));
            second.AsSpan(row * b, b).CopyTo(context.AsSpan(row * (a + b) + a));
        }

        return (context, rows);
    }

    /// <summary>Runs a text graph on one padded prompt of <paramref name="length"/> tokens (the mask input, if any, marks them).</summary>
    private (float[]? Hidden, float[]? Pooled) RunTextEncoder(string role, string graph, int[] tokens, int length, ImageRows? imageRows = null)
    {
        var encoder = Compiled(role, graph, null);
        encoder.SetInput<int>(encoder.Graph.Inputs.Single(i => i.Name == DiffusionGraphs.Tokens), tokens);
        if (encoder.Graph.Inputs.FirstOrDefault(i => i.Name == "positions") is { } positionsInput)
        {
            // A vision-language text encoder: M-RoPE positions, and the image rows' embeddings (none: the rows' own indices, no image rows).
            int axes = (int)positionsInput.Shape[1].Fixed;
            var rowPositions = imageRows?.Positions;
            if (rowPositions is null)
            {
                rowPositions = new int[tokens.Length * axes];
                for (int row = 0; row < tokens.Length; row++)
                {
                    rowPositions.AsSpan(row * axes, Math.Min(axes, 3)).Fill(row);
                }
            }

            encoder.SetInput<int>(positionsInput, rowPositions);
            encoder.SetInput<float>(encoder.Graph.Inputs.Single(i => i.Name == "embeddings"), imageRows?.Embeddings ?? new float[tokens.Length * (int)encoder.Graph.Inputs.Single(i => i.Name == "embeddings").Shape[1].Fixed]);
            encoder.SetInput<float>(encoder.Graph.Inputs.Single(i => i.Name == "embedding_mask"), imageRows?.Mask ?? new float[tokens.Length]);
        }

        if (encoder.Graph.Inputs.FirstOrDefault(i => i.Name == DiffusionGraphs.TokenMask) is { } maskInput)
        {
            var mask = new float[tokens.Length];
            mask.AsSpan(0, Math.Min(length, tokens.Length)).Fill(1f);
            encoder.SetInput<float>(maskInput, mask);
        }

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

    /// <summary>A prompt's padded ids (<see cref="PromptTokenizer"/>) and the number of prompt tokens before the padding.</summary>
    private static (int[] Ids, int Length) PromptTokens(ITokenizer tokenizer, int contextLength, string prompt, string? template = null)
    {
        int[] ids = PromptTokenizer.Tokenize(tokenizer, contextLength, prompt, out int count, template);
        return (ids, count);
    }

    private float[] Denoise(DiffusionRun run, PromptConditioning conditioning, DiffusionSampler sampler, int index, float[] latents, float[]? references, float[]? condition)
    {
        var denoiser = Compiled(ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, run.Dimensions);
        int rows = run.Rows;
        var batchDim = Input(denoiser, DiffusionGraphs.Latent).Shape[0];
        // Three-row steps (image guidance) run as two rows and one row: the Vulkan UNet kernels are exact for batches of one and two only.
        int perRun = batchDim.IsSymbolic ? Math.Min(rows, 2) : (int)batchDim.Fixed;
        var output = new float[run.LatentSize * rows];
        if (perRun >= rows)
        {
            RunDenoiser(denoiser, run, conditioning, sampler, index, latents, references, condition, 0, rows, output);
        }
        else
        {
            // A denoiser built for fewer rows (a fixed batch of one) runs once per row of a guided step.
            for (int row = 0; row < rows; row += perRun)
            {
                int count = Math.Min(perRun, rows - row);
                RunDenoiser(denoiser, run, conditioning, sampler, index, latents, references, condition, row, count, output.AsSpan(row * run.LatentSize, count * run.LatentSize));
            }
        }

        if (!run.Guided)
        {
            return output;
        }

        var guided = new float[run.LatentSize];
        if (rows == 3)
        {
            // InstructPix2Pix: rows are the instruction with the image, the empty prompt with the image, the empty prompt without it.
            for (int i = 0; i < guided.Length; i++)
            {
                float text = output[i];
                float image = output[run.LatentSize + i];
                float neither = output[2 * run.LatentSize + i];
                guided[i] = neither + run.Guidance * (text - image) + run.ImageGuidance * (image - neither);
            }

            return guided;
        }

        for (int i = 0; i < guided.Length; i++)
        {
            float unconditional = output[i];
            float conditional = output[run.LatentSize + i];
            guided[i] = unconditional + run.Guidance * (conditional - unconditional);
        }

        if (_guidanceRenormalization)
        {
            RenormalizePerToken(guided, output.AsSpan(run.LatentSize, run.LatentSize), run);
        }

        return guided;
    }

    /// <summary>
    /// Scales the guided prediction of every token (a patch of <see cref="_patchSize"/>² latent pixels over all channels) to the norm
    /// of the conditional prediction of that token, as diffusers' Qwen-Image pipelines do on their packed latents.
    /// </summary>
    private void RenormalizePerToken(float[] guided, ReadOnlySpan<float> conditional, DiffusionRun run)
    {
        int p = _patchSize;
        int rows = run.LatentHeight / p;
        int columns = run.LatentWidth / p;
        int plane = run.LatentHeight * run.LatentWidth;
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                double conditionalSquares = 0;
                double guidedSquares = 0;
                for (int c = 0; c < run.LatentChannels; c++)
                {
                    for (int dy = 0; dy < p; dy++)
                    {
                        int offset = c * plane + (row * p + dy) * run.LatentWidth + column * p;
                        for (int dx = 0; dx < p; dx++)
                        {
                            conditionalSquares += conditional[offset + dx] * (double)conditional[offset + dx];
                            guidedSquares += guided[offset + dx] * (double)guided[offset + dx];
                        }
                    }
                }

                float scale = guidedSquares > 0 ? (float)(Math.Sqrt(conditionalSquares) / Math.Sqrt(guidedSquares)) : 1f;
                for (int c = 0; c < run.LatentChannels; c++)
                {
                    for (int dy = 0; dy < p; dy++)
                    {
                        int offset = c * plane + (row * p + dy) * run.LatentWidth + column * p;
                        for (int dx = 0; dx < p; dx++)
                        {
                            guided[offset + dx] *= scale;
                        }
                    }
                }
            }
        }
    }

    /// <summary>One denoiser run over <paramref name="rows"/> conditioning rows starting at <paramref name="firstRow"/>.</summary>
    private static void RunDenoiser(CompiledModel denoiser, DiffusionRun run, PromptConditioning conditioning, DiffusionSampler sampler, int index,
        float[] latents, float[]? references, float[]? condition, int firstRow, int rows, Span<float> output)
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

        int totalRows = run.Rows;
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

        if (denoiser.Graph.Inputs.Any(i => i.Name == DiffusionGraphs.ContextMask))
        {
            var mask = conditioning.Mask ?? throw new InvalidOperationException("The denoiser needs a context mask, which the conditioning lacks.");
            int maskRow = mask.Length / totalRows;
            denoiser.SetInput<float>(Input(denoiser, DiffusionGraphs.ContextMask), mask.AsSpan(firstRow * maskRow, rows * maskRow).ToArray());
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

        if (denoiser.Graph.Inputs.Any(i => i.Name == DiffusionGraphs.ConditionLatent))
        {
            var extra = condition ?? throw new InvalidOperationException("The denoiser takes extra input channels, but the request gave no image.");
            var rowsOfCondition = new float[extra.Length * rows];
            for (int row = 0; row < rows; row++)
            {
                // The image-guidance row without the image has zeros as its extra channels.
                if (!(run.ImageGuidance > 0 && firstRow + row == 2))
                {
                    extra.CopyTo(rowsOfCondition, row * extra.Length);
                }
            }

            denoiser.SetInput<float>(Input(denoiser, DiffusionGraphs.ConditionLatent), rowsOfCondition);
        }

        if (denoiser.Graph.Inputs.Any(i => i.Name == DiffusionGraphs.ReferenceLatent))
        {
            denoiser.SetInput<float>(Input(denoiser, DiffusionGraphs.ReferenceLatent),
                references ?? throw new InvalidOperationException("The editing denoiser was built for reference images, but the request has none."));
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
    private float[] EncodeImage(ImageInput image, DiffusionRun run, bool lanczos = false)
    {
        var planes = new float[3 * image.Width * image.Height];
        PlanarImage.ToPlanar(image, planes, 0f, 1f);
        var resized = PlanarResize.Resize(planes, 3, image.Height, image.Width, run.Height, run.Width, lanczos);
        var encoder = Compiled(ModelComponentRoles.Vae, DiffusionGraphs.VaeEncode, run.Dimensions);
        encoder.SetInput<float>(Input(encoder, DiffusionGraphs.Image), resized);
        var parameters = new RunParameters();
        parameters.Symbols[DiffusionGraphs.BatchSymbol] = 1;
        encoder.Run(parameters);
        var latents = new float[run.LatentSize];
        encoder.GetOutput<float>(DiffusionGraphs.Latent, latents);
        return latents;
    }

    /// <summary>
    /// The extra input channels of a UNet with <see cref="DiffusionGraphs.ConditionLatent"/>: the image's unscaled latents
    /// (InstructPix2Pix), or the latent mask (1 = regenerate) followed by the latents of the image with the masked pixels grey (inpainting).
    /// </summary>
    private float[] ConditionLatents(ImageGenerationRequest request, DiffusionRun run, float[]? mask)
    {
        var image = request.Image!;
        if (_conditionMode == DiffusionConfigKeys.ConditionImage)
        {
            var latents = EncodeImage(image, run);
            for (int i = 0; i < latents.Length; i++)
            {
                latents[i] /= _scalingFactor;
            }

            return latents;
        }

        var planes = new float[3 * image.Width * image.Height];
        PlanarImage.ToPlanar(image, planes, 0f, 1f);
        var resized = PlanarResize.Resize(planes, 3, image.Height, image.Width, run.Height, run.Width);
        if (request.Mask is { } pixelMask)
        {
            // diffusers: masked_image = image * (mask < 0.5) in [-1, 1], that is mid grey inside the mask.
            var maskPlanes = new float[3 * pixelMask.Width * pixelMask.Height];
            PlanarImage.ToPlanar(pixelMask, maskPlanes, 0f, 1f);
            var maskResized = PlanarResize.Resize(maskPlanes, 3, pixelMask.Height, pixelMask.Width, run.Height, run.Width);
            int pixels = run.Width * run.Height;
            for (int i = 0; i < pixels; i++)
            {
                if ((maskResized[i] + maskResized[pixels + i] + maskResized[2 * pixels + i]) / 3f >= 0.5f)
                {
                    resized[i] = resized[pixels + i] = resized[2 * pixels + i] = 0.5f;
                }
            }
        }
        else
        {
            Array.Fill(resized, 0.5f);
        }

        var encoder = Compiled(ModelComponentRoles.Vae, DiffusionGraphs.VaeEncode, run.Dimensions);
        encoder.SetInput<float>(Input(encoder, DiffusionGraphs.Image), resized);
        var parameters = new RunParameters();
        parameters.Symbols[DiffusionGraphs.BatchSymbol] = 1;
        encoder.Run(parameters);
        int plane = run.LatentWidth * run.LatentHeight;
        var condition = new float[plane + run.LatentSize];
        if (mask is null)
        {
            Array.Fill(condition, 1f, 0, plane);
        }
        else
        {
            mask.CopyTo(condition, 0);
        }

        encoder.GetOutput<float>(DiffusionGraphs.Latent, condition.AsSpan(plane));
        return condition;
    }

    /// <summary>The model latents of the reference images (each resized to the output size) for an editing denoiser, [R, C, h, w].</summary>
    private float[] EncodeReferences(IReadOnlyList<ImageInput> images, DiffusionRun run)
    {
        var latents = new float[run.LatentSize * images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            // Qwen-Image-Edit's pipeline resizes with Lanczos (diffusers' VaeImageProcessor), as for the language model's view.
            EncodeImage(images[i], run, lanczos: _vision is not null).CopyTo(latents, i * run.LatentSize);
        }

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
        string key = dimensions is null ? graph
            : $"{graph}@{dimensions[DiffusionGraphs.Width]}x{dimensions[DiffusionGraphs.Height]}+{dimensions.GetValueOrDefault(DiffusionGraphs.References)}";
        if (!_compiled.TryGetValue(key, out var compiled))
        {
            try
            {
                compiled = _model.CompileComponent(role, graph, null, dimensions);
            }
            catch (InsufficientMemoryException) when (_compiled.Count > 0)
            {
                // The graphs held so far take more device memory than the weights' file sizes suggested (weights the backend widens,
                // other processes on the device): release the other stage (the text encoders, or the denoiser and VAE) and from now
                // on alternate, as for models that barely fit.
                bool text = IsTextGraph(graph);
                foreach (string other in _compiled.Keys.Where(k => IsTextGraph(k) != text).ToList())
                {
                    Release(other);
                }

                _releaseTextEncoders = true;
                compiled = _model.CompileComponent(role, graph, null, dimensions);
            }

            _compiled[key] = compiled;
        }

        return compiled;
    }

    private static bool IsTextGraph(string key)
        => key is DiffusionGraphs.TextEncode or DiffusionGraphs.TextEncode2 or DiffusionGraphs.TextEncode3 or WellKnown.VisionEncodeGraph;

    private void Release(string graph)
    {
        if (_compiled.Remove(graph, out var compiled))
        {
            compiled.Dispose();
        }
    }

    private static Graph.Value Input(CompiledModel model, string name) => model.Graph.Inputs.Single(i => i.Name == name);

    /// <summary>The reference images as the vision-language text encoder sees them: preprocessed, and the vision tower's embeddings per image.</summary>
    private sealed record ImagePrompt(IReadOnlyList<PreprocessedImage> Images, IReadOnlyList<float[]> Embeddings);

    /// <summary>The image inputs of the text graph: M-RoPE positions [rows, axes], the rows' embeddings [rows, width] and their mask [rows].</summary>
    private sealed record ImageRows(int[] Positions, float[] Embeddings, float[] Mask);
}
