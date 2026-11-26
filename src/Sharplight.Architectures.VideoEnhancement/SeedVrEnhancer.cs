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
/// precomputed text embedding, decoded, colour-corrected against the resized input and cropped back.
/// </summary>
/// <remarks>
/// Streaming: the frames are read batch by batch (one batch read ahead) and a group of batches is encoded, restored and decoded before the
/// next group is read; finished frames leave through a short queue that holds the enhancer back when the consumer falls behind. Memory is
/// that of a group, whatever the clip's length. A group is one batch while the networks fit the device together; when the device runs out
/// of memory the enhancer releases its compiled graphs (the DiT's weights and the VAE's activations need not fit together) and compiles them
/// again from the cached kernels, and then doubles the group (up to <see cref="MaxGroupBatches"/> and <see cref="GroupInputBudgetBytes"/>)
/// so the stages take turns once per group rather than once per batch. Batches are independent (each has its own noise streams), so the
/// grouping does not change the result. Frame sizes whose autoencoder activations do not fit are encoded and decoded in overlapping
/// tiles (<see cref="SpatialTiling"/>), chosen from the device's free memory and what the compiled graphs took, and made smaller after an
/// out-of-memory error.
/// </remarks>
internal sealed class SeedVrEnhancer : IVideoEnhancer
{
    /// <summary>The noise of batch b is stream 2b, its encoder sample stream 2b + 1.</summary>
    private const int StreamsPerBatch = 2;

    /// <summary>Updates waiting for the consumer before the enhancer waits (a few batches of finished frames at most).</summary>
    private const int QueuedUpdates = 4;

    /// <summary>The most batches encoded, restored and decoded as one group.</summary>
    private const int MaxGroupBatches = 8;

    /// <summary>The most bytes of input frames a group holds.</summary>
    private const long GroupInputBudgetBytes = 1L << 30;

    /// <summary>The longest batch of a whole-clip request when the model does not say (frames).</summary>
    private const int DefaultMaxFramesPerBatch = 65;

    /// <summary>The least overlap of neighbouring autoencoder tiles, in output pixels.</summary>
    private const int TileOverlap = 64;

    /// <summary>The shortest tile side the automatic tiling goes down to, in output pixels.</summary>
    private const int MinTileSide = 128;

    /// <summary>The share of the free device memory an automatically sized tile may take.</summary>
    private const double TileMemoryShare = 0.85;

    private readonly TaskModel _model;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Dictionary<string, CompiledModel> _compiled = new(StringComparer.Ordinal);
    private readonly Dictionary<string, TileLayout> _tiles = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (long Area, double Bytes)> _bytesPerPixel = new(StringComparer.Ordinal);
    private readonly float _timestep;
    private readonly int _latentChannels;
    private readonly int _spatialDownscale;
    private int _releases;
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
        Defaults = new VideoEnhancementDefaults(
            (float)description.ConfigValue(SeedVrConfigKeys.DefaultScale, 2.0),
            description.ConfigValue(SeedVrConfigKeys.SizeMultiple, 16),
            description.ConfigValue(SeedVrConfigKeys.FrameMultiple, 4),
            ParseColorCorrection(description.ConfigValue(SeedVrConfigKeys.ColorCorrection, "wavelet")),
            0,
            description.ConfigValue(SeedVrConfigKeys.MaxFramesPerBatch, DefaultMaxFramesPerBatch));
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public VideoEnhancementDefaults Defaults { get; }

    /// <summary>The tile layouts chosen so far, by graph and padded size (for tests and logs).</summary>
    internal IReadOnlyDictionary<string, TileLayout> TileLayouts => _tiles;

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        return serviceType.IsInstanceOfType(_model) ? _model
            : serviceType.IsInstanceOfType(this) ? this
            : null;
    }

    /// <inheritdoc/>
    public IAsyncEnumerable<VideoEnhancementUpdate> EnhanceStreamingAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        var settings = options ?? VideoEnhancementOptions.Default;
        var frames = request.ReadFramesAsync(cancellationToken);
        if (request.Frames is { } list)
        {
            // Frames in memory are checked before anything runs, as they always were; a source is checked frame by frame as it is read.
            Validate(list, settings);
        }

        return BackgroundCompute.StreamBlocking<VideoEnhancementUpdate>((emit, token) => Run(request, frames, settings, emit, token), QueuedUpdates,
            cancellationToken);
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

    private void Validate(IReadOnlyList<ImageInput> frames, VideoEnhancementOptions options)
    {
        if (frames.Count == 0)
        {
            throw new ArgumentException("A video-enhancement request needs at least one frame.", nameof(frames));
        }

        for (int i = 0; i < frames.Count; i++)
        {
            EnhancementPlan.CheckFrame(frames[i], frames[0].Width, frames[0].Height, i);
        }

        var plan = EnhancementPlan.Create(frames[0], frames.Count, null, options, Defaults);
        if ((options.InitialNoise is not null || options.EncoderNoise is not null) && plan.ExpectedBatches > 1)
        {
            throw new ArgumentException("Explicit noise needs the whole clip in one batch (FramesPerBatch = 0).", nameof(options));
        }
    }

    private void Run(VideoEnhancementRequest request, IAsyncEnumerable<ImageInput> frames, VideoEnhancementOptions options, Action<VideoEnhancementUpdate> emit,
        CancellationToken cancellationToken)
    {
        _gate.Wait(cancellationToken);
        try
        {
            using var reader = new FrameBatchReader(frames, cancellationToken);
            var first = reader.PeekFirst() ?? throw new ArgumentException("A video-enhancement request needs at least one frame.", nameof(request));
            var plan = EnhancementPlan.Create(first, request.KnownFrameCount, request.FramesPerSecond, options, Defaults);
            var progress = new VideoEnhancementUpdate(VideoEnhancementPhase.Encoding, 0, plan.ExpectedBatches)
            {
                TotalFrames = plan.TotalFrames,
                Seed = plan.Seed,
                FramesPerSecond = plan.FramesPerSecond,
                Size = (plan.Width, plan.Height),
            };
            int groupSize = 1;
            int maxGroup = MaxGroup(plan);
            int batches = 0;
            long read = 0;
            long finished = 0;
            bool ended = false;
            while (!ended)
            {
                var group = new List<(EnhancementBatch Batch, List<ImageInput> Frames)>(groupSize);
                while (group.Count < groupSize && !ended)
                {
                    var batchFrames = reader.Take(plan.FramesPerBatch);
                    ended = batchFrames.Count < plan.FramesPerBatch;
                    if (batchFrames.Count == 0)
                    {
                        break;
                    }

                    for (int i = 0; i < batchFrames.Count; i++)
                    {
                        EnhancementPlan.CheckFrame(batchFrames[i], plan.InputWidth, plan.InputHeight, read + i);
                    }

                    if (batches > 0 && (options.InitialNoise is not null || options.EncoderNoise is not null))
                    {
                        throw new ArgumentException("Explicit noise needs the whole clip in one batch (FramesPerBatch = 0).", nameof(options));
                    }

                    group.Add((plan.Batch(batches, read, batchFrames.Count), batchFrames));
                    batches++;
                    read += batchFrames.Count;
                }

                if (group.Count == 0)
                {
                    break;
                }

                int total = ended ? batches : Math.Max(plan.ExpectedBatches, batches);
                int releases = _releases;
                // A group of one keeps its resized clip for the colour correction; larger groups prepare it again instead of holding every batch's pixels.
                float[]? kept = null;
                var conditions = new float[group.Count][];
                for (int g = 0; g < group.Count; g++)
                {
                    var (batch, input) = group[g];
                    cancellationToken.ThrowIfCancellationRequested();
                    emit(progress with { Phase = VideoEnhancementPhase.Encoding, Batch = batch.Index, TotalBatches = total, FirstFrame = batch.Start, FinishedFrames = finished });
                    var clip = Prepare(plan, batch, input);
                    conditions[g] = Encode(plan, batch, clip, batch.Index == 0 ? options.EncoderNoise : null);
                    kept = group.Count == 1 ? clip : null;
                }

                var restored = new float[group.Count][];
                for (int g = 0; g < group.Count; g++)
                {
                    var batch = group[g].Batch;
                    cancellationToken.ThrowIfCancellationRequested();
                    emit(progress with { Phase = VideoEnhancementPhase.Restoring, Batch = batch.Index, TotalBatches = total, FirstFrame = batch.Start, FinishedFrames = finished });
                    restored[g] = Restore(plan, batch, conditions[g], batch.Index == 0 ? options.InitialNoise : null);
                    conditions[g] = [];
                }

                for (int g = 0; g < group.Count; g++)
                {
                    var (batch, input) = group[g];
                    cancellationToken.ThrowIfCancellationRequested();
                    emit(progress with { Phase = VideoEnhancementPhase.Decoding, Batch = batch.Index, TotalBatches = total, FirstFrame = batch.Start, FinishedFrames = finished });
                    var images = Decode(plan, batch, restored[g], kept ?? Prepare(plan, batch, input));
                    restored[g] = [];
                    kept = null;
                    group[g] = (batch, []);
                    finished += images.Count;
                    emit(progress with
                    {
                        Phase = VideoEnhancementPhase.Decoding,
                        Batch = batch.Index,
                        TotalBatches = total,
                        FirstFrame = batch.Start,
                        FinishedFrames = finished,
                        Frames = images,
                    });
                }

                if (_releases != releases && groupSize < maxGroup)
                {
                    groupSize = Math.Min(maxGroup, groupSize * 2);
                    _model.Options.Log?.Invoke($"video enhancement: the stages take turns on the device; {groupSize} batches per group from now on");
                }
            }

            emit(progress with
            {
                Phase = VideoEnhancementPhase.Completed,
                Batch = batches - 1,
                TotalBatches = batches,
                FirstFrame = finished,
                FinishedFrames = finished,
                TotalFrames = finished,
            });
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>The most batches a group may hold: <see cref="MaxGroupBatches"/>, fewer when their input frames would pass <see cref="GroupInputBudgetBytes"/>.</summary>
    private static int MaxGroup(EnhancementPlan plan)
    {
        if (plan.FramesPerBatch == int.MaxValue)
        {
            return 1;
        }

        long batchBytes = 3L * plan.InputWidth * plan.InputHeight * plan.FramesPerBatch;
        return (int)Math.Clamp(GroupInputBudgetBytes / Math.Max(1, batchBytes), 1, MaxGroupBatches);
    }

    /// <summary>The frames of a batch resized and padded: [frames, 3, H, W] in [−1, 1].</summary>
    private static float[] Prepare(EnhancementPlan plan, EnhancementBatch batch, List<ImageInput> frames)
        => ClipPixels.Prepare(frames, plan.Width, plan.Height, plan.PaddedWidth, plan.PaddedHeight, batch.PaddedFrames);

    /// <summary>Encodes one batch and samples the latent: mean + deviation · noise (scaled latent space, [C, T, h, w]).</summary>
    private float[] Encode(EnhancementPlan plan, EnhancementBatch batch, float[] clip, float[]? explicitNoise)
    {
        int size = LatentSize(plan, batch);
        int latentPlanes = _latentChannels * batch.LatentFrames;
        // Mean and deviation as one stack of planes, so the tiles of both are blended alike.
        var distribution = Tiled(SeedVrGraphs.VaeEncode, plan, tiles => SpatialTiling.Run(clip, batch.PaddedFrames * 3, plan.PaddedHeight, plan.PaddedWidth,
            tiles.Rows, tiles.Columns, 1, _spatialDownscale, int.MaxValue, (tile, height, width) =>
            {
                int tileSize = latentPlanes * (height / _spatialDownscale) * (width / _spatialDownscale);
                var both = new float[2 * tileSize];
                RunGraph(ModelComponentRoles.Vae, SeedVrGraphs.VaeEncode, Dimensions(batch, height, width),
                    encoder => encoder.SetInput<float>(Input(encoder, VideoDiffusionGraphs.Frames), tile),
                    encoder =>
                    {
                        encoder.GetOutput<float>(SeedVrGraphs.Mean, both.AsSpan(0, tileSize));
                        encoder.GetOutput<float>(SeedVrGraphs.Deviation, both.AsSpan(tileSize));
                    });
                return both;
            }));
        var mean = distribution.AsSpan(0, size);
        var deviation = distribution.AsSpan(size, size);
        var noise = explicitNoise?.AsSpan(0, size).ToArray() ?? LatentNoise.Draw(plan.Seed, StreamsPerBatch * batch.Index + 1, size);
        var latent = new float[size];
        for (int i = 0; i < size; i++)
        {
            latent[i] = mean[i] + deviation[i] * noise[i];
        }

        return latent;
    }

    /// <summary>The one-step restoration: x₀ = x_t − t/1000 · v from noise x_t at the model's timestep.</summary>
    private float[] Restore(EnhancementPlan plan, EnhancementBatch batch, float[] condition, float[]? explicitNoise)
    {
        int size = LatentSize(plan, batch);
        var latent = explicitNoise?.AsSpan(0, size).ToArray() ?? LatentNoise.Draw(plan.Seed, StreamsPerBatch * batch.Index, size);
        var prediction = new float[size];
        RunGraph(ModelComponentRoles.Transformer, SeedVrGraphs.Restore, Dimensions(batch, plan.PaddedHeight, plan.PaddedWidth),
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
        int latentHeight = plan.PaddedHeight / _spatialDownscale;
        int latentWidth = plan.PaddedWidth / _spatialDownscale;
        var decoded = Tiled(SeedVrGraphs.VaeDecode, plan, tiles => SpatialTiling.Run(latent, _latentChannels * batch.LatentFrames, latentHeight, latentWidth,
            InLatents(tiles.Rows), InLatents(tiles.Columns), _spatialDownscale, 1, int.MaxValue, (tile, height, width) =>
            {
                var pixels = new float[batch.PaddedFrames * 3 * height * _spatialDownscale * width * _spatialDownscale];
                RunGraph(ModelComponentRoles.Vae, SeedVrGraphs.VaeDecode, Dimensions(batch, height * _spatialDownscale, width * _spatialDownscale),
                    decoder => decoder.SetInput<float>(Input(decoder, VideoDiffusionGraphs.Latent), tile),
                    decoder => decoder.GetOutput<float>(VideoDiffusionGraphs.Frames, pixels));
                return pixels;
            }));
        switch (plan.ColorCorrection)
        {
            case VideoColorCorrection.Wavelet:
                RunGraph(ModelComponentRoles.Vae, SeedVrGraphs.ColorCorrect, Dimensions(batch, plan.PaddedHeight, plan.PaddedWidth),
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

    /// <summary>
    /// Runs an autoencoder stage over the padded frame in the tiles chosen for its graph and size (<see cref="ChooseTiles"/>; the whole
    /// frame when it fits). When the device runs out of memory even after the compiled graphs were released, the tiles are made smaller
    /// (half the area, down to <see cref="MinTileSide"/>) and the stage runs again; the layout that worked is kept for the next batches.
    /// </summary>
    private float[] Tiled(string graph, EnhancementPlan plan, Func<TileLayout, float[]> run)
    {
        string key = $"{graph}@{plan.PaddedWidth}x{plan.PaddedHeight}";
        var tiles = ChooseTiles(graph, key, plan);
        while (true)
        {
            try
            {
                var result = run(tiles);
                _tiles[key] = tiles;
                return result;
            }
            catch (Exception error) when (error is DeviceOutOfMemoryException or InsufficientMemoryException && Smaller(tiles, plan) is { } smaller)
            {
                _model.Options.Log?.Invoke($"{key}: {error.GetType().Name} at {tiles}; {smaller} from now on");
                ReleaseAll();
                _releases++;
                tiles = smaller;
            }
        }
    }

    /// <summary>
    /// The tiles of a graph at a padded size: the layout that worked before; else the request's <see cref="VideoEnhancementOptions.TileSize"/>;
    /// else, once a compiled graph has shown how much device memory a pixel of this graph takes, the largest tiles that fit the free memory
    /// (with the memory of the graphs that would be released); else the whole frame.
    /// </summary>
    private TileLayout ChooseTiles(string graph, string key, EnhancementPlan plan)
    {
        if (plan.TileSize is { } side)
        {
            return TileLayout.WithSide(plan.PaddedHeight, plan.PaddedWidth, side, TileOverlap, Defaults.SizeMultiple);
        }

        if (_tiles.TryGetValue(key, out var known))
        {
            return known;
        }

        var whole = TileLayout.Whole(plan.PaddedHeight, plan.PaddedWidth);
        if (!_bytesPerPixel.TryGetValue(graph, out var measured) || _model.Options.Devices.Count == 0)
        {
            return whole;
        }

        double bytesPerPixel = measured.Bytes;
        var slot = _model.Options.Devices[0];
        long free = slot.MemoryLimit is { } limit ? limit - _compiled.Values.Sum(c => c.ResidentMemoryBytes) : slot.Device.AvailableMemory;
        double budget = TileMemoryShare * (free + _compiled.Values.Sum(c => c.ResidentMemoryBytes));
        long area = (long)plan.PaddedHeight * plan.PaddedWidth;
        if (bytesPerPixel * area <= budget)
        {
            return whole;
        }

        var tiles = TileLayout.Fit(plan.PaddedHeight, plan.PaddedWidth, (long)Math.Max(1, budget / bytesPerPixel), TileOverlap, Defaults.SizeMultiple, MinTileSide);
        _model.Options.Log?.Invoke($"{key}: ~{bytesPerPixel * area / (1 << 30):F1} GiB for the whole frame, {budget / (1 << 30):F1} GiB free; {tiles}");
        return tiles;
    }

    /// <summary>Tiles of about half the area of <paramref name="tiles"/>, or <see langword="null"/> when they are as small as they go.</summary>
    private TileLayout? Smaller(TileLayout tiles, EnhancementPlan plan)
    {
        var smaller = TileLayout.Fit(plan.PaddedHeight, plan.PaddedWidth, Math.Max(1, tiles.TileArea / 2), TileOverlap, Defaults.SizeMultiple, MinTileSide);
        return smaller.TileArea < tiles.TileArea ? smaller : null;
    }

    /// <summary>Pixel tiles as latent tiles.</summary>
    private IReadOnlyList<TileSpan> InLatents(IReadOnlyList<TileSpan> spans)
        => [.. spans.Select(span => new TileSpan(span.Start / _spatialDownscale, span.Length / _spatialDownscale))];

    private int LatentSize(EnhancementPlan plan, EnhancementBatch batch)
        => _latentChannels * batch.LatentFrames * (plan.PaddedHeight / _spatialDownscale) * (plan.PaddedWidth / _spatialDownscale);

    private Dictionary<string, long> Dimensions(EnhancementBatch batch, int height, int width) => new(StringComparer.Ordinal)
    {
        [VideoDiffusionGraphs.FramesDimension] = batch.LatentFrames,
        [VideoDiffusionGraphs.Height] = height,
        [VideoDiffusionGraphs.Width] = width,
        [SeedVrGraphs.HalfCaches] = _model.Options.AllowReducedPrecision ? 1 : 0,
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
                    _model.Options.Log?.Invoke($"{key}: {compiled.ResidentMemoryBytes / (1024.0 * 1024 * 1024):F2} GiB on the device");
                    // What a pixel of this graph takes (weights included, so an overestimate for small sizes), for choosing tiles of other sizes.
                    // The largest size seen gives the best figure (the weights matter least there).
                    long area = dimensions[VideoDiffusionGraphs.Width] * dimensions[VideoDiffusionGraphs.Height];
                    if (!_bytesPerPixel.TryGetValue(graph, out var seen) || area >= seen.Area)
                    {
                        _bytesPerPixel[graph] = (area, compiled.ResidentMemoryBytes / (double)area);
                    }
                }

                bind(compiled);
                compiled.Run(new RunParameters());
                read(compiled);
                return;
            }
            catch (Exception error) when (attempt == 0 && _compiled.Count > 0 && error is DeviceOutOfMemoryException or InsufficientMemoryException)
            {
                // An allocation failed, or placement found no room for the weights next to the graphs already compiled.
                _model.Options.Log?.Invoke($"{graph}: {error.GetType().Name}; releasing {_compiled.Count} compiled graph(s) and retrying");
                ReleaseAll();
                _releases++;
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
