using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.VideoEnhancement;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>
/// The <see cref="IVideoEnhancer"/> of SeedVR2 models (the official inference path of projects/inference_seedvr2_*.py): each batch of frames
/// is resized (antialiased bicubic) to the output size, padded to the network's multiples, encoded by the causal VAE (the latent sampled
/// from its distribution), restored in one step from noise at the model's timestep by the DiT conditioned on that latent and the model's
/// precomputed text embedding, decoded, colour-corrected against the resized input and cropped back. All batches are encoded first, then
/// restored, then decoded; compiled graphs stay for later requests unless the device runs out of memory, which releases them, so only one
/// network's weights and activations need to fit at a time.
/// </summary>
internal sealed class SeedVrEnhancer : IVideoEnhancer
{
    /// <summary>The noise of batch b is stream 2b, its encoder sample stream 2b + 1.</summary>
    private const int StreamsPerBatch = 2;

    private readonly TaskModel _model;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Dictionary<string, CompiledModel> _compiled = new(StringComparer.Ordinal);
    private readonly VideoEnhancementDefaults _defaults;
    private readonly float _timestep;
    private readonly int _latentChannels;
    private readonly int _spatialDownscale;
    private bool _disposed;

    /// <summary>Creates the enhancer for a loaded SeedVR2 model.</summary>
    /// <param name="model">The model.</param>
    public SeedVrEnhancer(TaskModel model)
    {
        _model = model;
        var description = model.Description;
        var vae = description.Component(ModelComponentRoles.Vae);
        _timestep = (float)description.ConfigValue(SeedVrConfigKeys.Timestep, 1000.0);
        _latentChannels = vae.ConfigValue(DiffusionConfigKeys.LatentChannels, 16);
        _spatialDownscale = vae.ConfigValue(SeedVrConfigKeys.SpatialDownscale, 8);
        _defaults = new VideoEnhancementDefaults(
            (float)description.ConfigValue(SeedVrConfigKeys.DefaultScale, 2.0),
            description.ConfigValue(SeedVrConfigKeys.SizeMultiple, 16),
            description.ConfigValue(SeedVrConfigKeys.FrameMultiple, 4),
            ParseColorCorrection(description.ConfigValue(SeedVrConfigKeys.ColorCorrection, "wavelet")),
            0);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    /// <remarks>Also answers <see cref="VideoEnhancementDefaults"/>.</remarks>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        return serviceType == typeof(VideoEnhancementDefaults) ? _defaults
            : serviceType.IsInstanceOfType(_model) ? _model
            : serviceType.IsInstanceOfType(this) ? this
            : null;
    }

    /// <inheritdoc/>
    public async Task<VideoEnhancementResult> EnhanceAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        VideoEnhancementResult? result = null;
        await foreach (var update in EnhanceStreamingAsync(request, options, cancellationToken).ConfigureAwait(false))
        {
            result = update.Result ?? result;
        }

        return result ?? throw new InvalidOperationException("Video enhancement produced no result.");
    }

    /// <inheritdoc/>
    public IAsyncEnumerable<VideoEnhancementUpdate> EnhanceStreamingAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        var plan = EnhancementPlan.Create(request, options ?? VideoEnhancementOptions.Default, _defaults);
        var settings = options ?? VideoEnhancementOptions.Default;
        if ((settings.InitialNoise is not null || settings.EncoderNoise is not null) && plan.Batches.Count > 1)
        {
            throw new ArgumentException("Explicit noise needs the whole clip in one batch (FramesPerBatch = 0).", nameof(options));
        }

        return BackgroundCompute.StreamBlocking<VideoEnhancementUpdate>((emit, token) => Run(request, settings, plan, emit, token), cancellationToken);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ReleaseAll();
        _gate.Dispose();
    }

    private void Run(VideoEnhancementRequest request, VideoEnhancementOptions options, EnhancementPlan plan, Action<VideoEnhancementUpdate> emit,
        CancellationToken cancellationToken)
    {
        _gate.Wait(cancellationToken);
        try
        {
            int batches = plan.Batches.Count;
            // The resized clip is the encoder's input and the colour correction's reference; one batch keeps it, longer requests prepare it
            // again for decoding instead of holding every batch's pixels.
            float[]? kept = null;
            var conditions = new float[batches][];
            for (int b = 0; b < batches; b++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                emit(new VideoEnhancementUpdate(VideoEnhancementPhase.Encoding, b, batches));
                var clip = Prepare(request, plan, b);
                conditions[b] = Encode(plan, b, clip, b == 0 ? options.EncoderNoise : null);
                kept = batches == 1 ? clip : null;
            }

            var restored = new float[batches][];
            for (int b = 0; b < batches; b++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                emit(new VideoEnhancementUpdate(VideoEnhancementPhase.Restoring, b, batches));
                restored[b] = Restore(plan, b, conditions[b], b == 0 ? options.InitialNoise : null);
            }

            var all = new List<ImageInput>(request.Frames.Count);
            for (int b = 0; b < batches; b++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                emit(new VideoEnhancementUpdate(VideoEnhancementPhase.Decoding, b, batches));
                var images = Decode(plan, plan.Batches[b], restored[b], kept ?? Prepare(request, plan, b));
                restored[b] = [];
                all.AddRange(images);
                emit(new VideoEnhancementUpdate(VideoEnhancementPhase.Decoding, b, batches, images));
            }

            var result = new VideoEnhancementResult(all, plan.FramesPerSecond, plan.Seed);
            emit(new VideoEnhancementUpdate(VideoEnhancementPhase.Completed, batches - 1, batches, Result: result));
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>The frames of batch <paramref name="index"/> resized and padded: [frames, 3, H, W] in [−1, 1].</summary>
    private static float[] Prepare(VideoEnhancementRequest request, EnhancementPlan plan, int index)
    {
        var batch = plan.Batches[index];
        var frames = request.Frames.Skip(batch.Start).Take(batch.Count).ToList();
        return ClipPixels.Prepare(frames, plan.Width, plan.Height, plan.PaddedWidth, plan.PaddedHeight, batch.PaddedFrames);
    }

    /// <summary>Encodes one batch and samples the latent: mean + deviation · noise (scaled latent space, [C, T, h, w]).</summary>
    private float[] Encode(EnhancementPlan plan, int index, float[] clip, float[]? explicitNoise)
    {
        var batch = plan.Batches[index];
        int size = LatentSize(plan, batch);
        var mean = new float[size];
        var deviation = new float[size];
        RunGraph(ModelComponentRoles.Vae, SeedVrGraphs.VaeEncode, Dimensions(plan, batch),
            encoder => encoder.SetInput<float>(Input(encoder, VideoDiffusionGraphs.Frames), clip),
            encoder =>
            {
                encoder.GetOutput<float>(SeedVrGraphs.Mean, mean);
                encoder.GetOutput<float>(SeedVrGraphs.Deviation, deviation);
            });
        var noise = explicitNoise?.AsSpan(0, size).ToArray() ?? LatentNoise.Draw(plan.Seed, StreamsPerBatch * index + 1, size);
        for (int i = 0; i < size; i++)
        {
            mean[i] += deviation[i] * noise[i];
        }

        return mean;
    }

    /// <summary>The one-step restoration: x₀ = x_t − t/1000 · v from noise x_t at the model's timestep.</summary>
    private float[] Restore(EnhancementPlan plan, int index, float[] condition, float[]? explicitNoise)
    {
        var batch = plan.Batches[index];
        int size = LatentSize(plan, batch);
        var latent = explicitNoise?.AsSpan(0, size).ToArray() ?? LatentNoise.Draw(plan.Seed, StreamsPerBatch * index, size);
        var prediction = new float[size];
        RunGraph(ModelComponentRoles.Transformer, SeedVrGraphs.Restore, Dimensions(plan, batch),
            transformer =>
            {
                transformer.SetInput<float>(Input(transformer, VideoDiffusionGraphs.Latent), latent);
                transformer.SetInput<float>(Input(transformer, SeedVrGraphs.Condition), condition);
                transformer.SetInput<float>(Input(transformer, VideoDiffusionGraphs.Timestep), [_timestep]);
            },
            transformer => transformer.GetOutput<float>(VideoDiffusionGraphs.Prediction, prediction));
        float sigma = _timestep / 1000f;
        for (int i = 0; i < size; i++)
        {
            latent[i] -= sigma * prediction[i];
        }

        return latent;
    }

    /// <summary>Decodes one batch, corrects its colours against the resized input and crops it to the requested frames and size.</summary>
    private List<ImageInput> Decode(EnhancementPlan plan, EnhancementBatch batch, float[] latent, float[] reference)
    {
        var dimensions = Dimensions(plan, batch);
        var decoded = new float[reference.Length];
        RunGraph(ModelComponentRoles.Vae, SeedVrGraphs.VaeDecode, dimensions,
            decoder => decoder.SetInput<float>(Input(decoder, VideoDiffusionGraphs.Latent), latent),
            decoder => decoder.GetOutput<float>(VideoDiffusionGraphs.Frames, decoded));
        switch (plan.ColorCorrection)
        {
            case VideoColorCorrection.Wavelet:
                RunGraph(ModelComponentRoles.Vae, SeedVrGraphs.ColorCorrect, dimensions,
                    corrector =>
                    {
                        corrector.SetInput<float>(Input(corrector, VideoDiffusionGraphs.Frames), decoded);
                        corrector.SetInput<float>(Input(corrector, SeedVrGraphs.Reference), reference);
                    },
                    corrector => corrector.GetOutput<float>(SeedVrGraphs.Output, decoded));
                return ClipPixels.ToImages(decoded, batch.Count, plan.PaddedWidth, plan.PaddedHeight, plan.Width, plan.Height, signed: false);
            case VideoColorCorrection.AdaIn:
                AdaptiveInstanceNorm.Apply(decoded, reference, batch.PaddedFrames, plan.PaddedWidth * plan.PaddedHeight);
                break;
        }

        return ClipPixels.ToImages(decoded, batch.Count, plan.PaddedWidth, plan.PaddedHeight, plan.Width, plan.Height, signed: true);
    }

    private int LatentSize(EnhancementPlan plan, EnhancementBatch batch)
        => _latentChannels * batch.LatentFrames * (plan.PaddedHeight / _spatialDownscale) * (plan.PaddedWidth / _spatialDownscale);

    private static Dictionary<string, long> Dimensions(EnhancementPlan plan, EnhancementBatch batch) => new(StringComparer.Ordinal)
    {
        [VideoDiffusionGraphs.FramesDimension] = batch.LatentFrames,
        [VideoDiffusionGraphs.Height] = plan.PaddedHeight,
        [VideoDiffusionGraphs.Width] = plan.PaddedWidth,
    };

    /// <summary>
    /// Runs one graph: compiles it once per size (kept for later requests), binds its inputs, runs it and reads its outputs. The stages share
    /// the device: when it runs out of memory, every compiled graph is released (the DiT's weights and the VAE's activations need not fit
    /// together) and the graph is compiled again, from the cached kernels, and run once more.
    /// </summary>
    private void RunGraph(string role, string graph, IReadOnlyDictionary<string, long> dimensions, Action<CompiledModel> bind, Action<CompiledModel> read)
    {
        string key = $"{graph}@{dimensions[VideoDiffusionGraphs.Width]}x{dimensions[VideoDiffusionGraphs.Height]}x{dimensions[VideoDiffusionGraphs.FramesDimension]}";
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                if (!_compiled.TryGetValue(key, out var compiled))
                {
                    compiled = _model.CompileComponent(role, graph, null, dimensions);
                    _compiled[key] = compiled;
                }

                bind(compiled);
                compiled.Run(new RunParameters());
                read(compiled);
                return;
            }
            catch (Exception error) when (attempt == 0 && _compiled.Count > 0 && error is DeviceOutOfMemoryException or InsufficientMemoryException)
            {
                // An allocation failed, or placement found no room for the weights next to the graphs already compiled.
                ReleaseAll();
            }
        }
    }

    private void ReleaseAll()
    {
        foreach (var compiled in _compiled.Values)
        {
            compiled.Dispose();
        }

        _compiled.Clear();
    }

    private static Graph.Value Input(CompiledModel model, string name) => model.Graph.Inputs.Single(i => i.Name == name);

    private static VideoColorCorrection ParseColorCorrection(string name) => name switch
    {
        "none" => VideoColorCorrection.None,
        "adain" => VideoColorCorrection.AdaIn,
        _ => VideoColorCorrection.Wavelet,
    };
}
