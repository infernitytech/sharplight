using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting;

/// <summary>
/// The effective settings of one hosted model: its <see cref="HostedModelOptions"/> with every unset value taken from
/// <see cref="ModelHostingOptions.Defaults"/>. What the loaders (<see cref="Loaders.IHostedModelLoaderFactory"/>) and the placement
/// policy see.
/// </summary>
public sealed record HostedModelSettings
{
    /// <summary>The registered name.</summary>
    public required string Name { get; init; }

    /// <summary>Model file or directory.</summary>
    public required string Path { get; init; }

    /// <summary>What the model does.</summary>
    public required ModelKind Kind { get; init; }

    /// <summary>The task kind of a <see cref="ModelKind.Task"/> model, else null.</summary>
    public string? Task { get; init; }

    /// <summary>The device specification (<see cref="DeviceSpecifications"/>).</summary>
    public required string Device { get; init; }

    /// <summary>Context length per sequence.</summary>
    public required int ContextLength { get; init; }

    /// <summary>Maximum rows of one forward pass.</summary>
    public required int BatchTokens { get; init; }

    /// <summary>Concurrent sequences.</summary>
    public required int Slots { get; init; }

    /// <summary>Paged KV cache size in tokens, or null.</summary>
    public long? KvCacheTokens { get; init; }

    /// <summary>Element type of the KV cache, or null for the model's default.</summary>
    public string? KvCacheType { get; init; }

    /// <summary>Vision projector file, or null.</summary>
    public string? Projector { get; init; }

    /// <summary>Chat template file that replaces the model's own, or null.</summary>
    public string? ChatTemplate { get; init; }

    /// <summary>Further files by role.</summary>
    public IReadOnlyDictionary<string, string> Auxiliary { get; init; } = new Dictionary<string, string>();

    /// <summary>Loads on its first request.</summary>
    public bool Lazy { get; init; }

    /// <summary>Never evicted and never unloaded when idle.</summary>
    public bool Pinned { get; init; }

    /// <summary>Layers that do not fit on the GPU run on the CPU.</summary>
    public bool CpuOffload { get; init; }

    /// <summary>With <see cref="CpuOffload"/>: routed experts that do not fit host memory stay on disk.</summary>
    public bool ExpertsOnDisk { get; init; }

    /// <summary>Segment workers (<c>host:port</c>) that run the later pipeline stages on other machines; empty for none.</summary>
    public IReadOnlyList<string> Workers { get; init; } = [];

    /// <summary>With <see cref="ExpertsOnDisk"/>: bytes of the expert cache, or null to size it from the free host memory.</summary>
    public long? ExpertCacheBytes { get; init; }

    /// <summary>Reduced-precision kernels allowed.</summary>
    public bool ReducedPrecision { get; init; }

    /// <summary>The block-scaled FP8 mode where it applies (<see cref="Tasks.TaskLoadOptions.BlockScaledFp8"/>).</summary>
    public bool BlockScaledFp8 { get; init; } = true;

    /// <summary>Own idle timeout, or null for the host's.</summary>
    public TimeSpan? IdleTimeout { get; init; }

    /// <summary>Waiting requests before new ones are refused.</summary>
    public int MaxQueue { get; init; }

    /// <summary>Prompt rows per step and sequence (0 = the step's rows).</summary>
    public int PrefillChunk { get; init; }

    /// <summary>Prefix caching on.</summary>
    public bool PrefixCache { get; init; } = true;

    /// <summary>Recurrent-state snapshots for prefix caching, or null for the default.</summary>
    public int? KvStateSnapshots { get; init; }

    /// <summary>Token budget of one engine step, or null.</summary>
    public int? StepTokens { get; init; }

    /// <summary>Smallest token budget of the load-adaptive engine step, or null (the engine's default).</summary>
    public int? MinStepTokens { get; init; }

    /// <summary>When the floor of the adaptive step budget applies, or null (the engine's default).</summary>
    public Llm.Serving.StepFloorPolicy? StepFloor { get; init; }

    /// <summary>Rows per sequence holding a slot that the adaptive step budget allows, or null (the engine's default).</summary>
    public int? StepRowsPerSequence { get; init; }

    /// <summary>How a paged KV cache admits requests when it runs short, or null (the engine's default: queue).</summary>
    public Llm.Serving.KvAdmissionPolicy? KvAdmission { get; init; }

    /// <summary>Prompt-lookup draft length (0 = off).</summary>
    public int NgramDraftTokens { get; init; }

    /// <summary>Multi-token-prediction draft length (0 = off).</summary>
    public int MtpDraftTokens { get; init; }

    /// <summary>Speculate only while at most this many sequences decode.</summary>
    public int SpeculationMaxBatch { get; init; }

    /// <summary>Text shown in listings.</summary>
    public string? Description { get; init; }

    /// <summary>Merges <paramref name="model"/> with <paramref name="defaults"/>.</summary>
    /// <param name="model">The model's own settings.</param>
    /// <param name="defaults">The host's defaults.</param>
    /// <param name="lazyByDefault">Whether models without their own setting load lazily.</param>
    /// <returns>The effective settings.</returns>
    /// <exception cref="ArgumentException">The model has no name or no path.</exception>
    public static HostedModelSettings Resolve(HostedModelOptions model, HostedModelDefaults defaults, bool lazyByDefault)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(defaults);
        if (string.IsNullOrWhiteSpace(model.Name) || string.IsNullOrWhiteSpace(model.Path))
        {
            throw new ArgumentException($"Every hosted model needs a name and a path (got name '{model.Name}', path '{model.Path}').", nameof(model));
        }

        return new HostedModelSettings
        {
            Name = model.Name,
            Path = model.Path,
            Kind = model.Kind,
            Task = model.Task,
            Device = string.IsNullOrWhiteSpace(model.Device) ? defaults.Device : model.Device,
            ContextLength = model.ContextLength ?? defaults.ContextLength,
            BatchTokens = model.BatchTokens ?? defaults.BatchTokens,
            Slots = Math.Max(1, model.Slots ?? defaults.Slots),
            KvCacheTokens = model.KvCacheTokens ?? defaults.KvCacheTokens,
            KvCacheType = model.KvCacheType ?? defaults.KvCacheType,
            Projector = model.Projector ?? defaults.Projector,
            ChatTemplate = model.ChatTemplate ?? defaults.ChatTemplate,
            Auxiliary = model.Auxiliary is { } auxiliary ? new Dictionary<string, string>(auxiliary, StringComparer.Ordinal) : new Dictionary<string, string>(),
            Lazy = model.Lazy ?? lazyByDefault,
            Pinned = model.Pinned ?? false,
            CpuOffload = model.CpuOffload ?? defaults.CpuOffload,
            ExpertsOnDisk = model.ExpertsOnDisk ?? defaults.ExpertsOnDisk,
            Workers = model.Workers is { } workers ? [.. workers.Where(w => !string.IsNullOrWhiteSpace(w))] : [],
            ExpertCacheBytes = model.ExpertCacheGiB is > 0 ? (long)(model.ExpertCacheGiB.Value * (1L << 30)) : null,
            ReducedPrecision = defaults.ReducedPrecision,
            BlockScaledFp8 = model.BlockScaledFp8 ?? defaults.BlockScaledFp8,
            IdleTimeout = model.IdleTimeoutSeconds is > 0 ? TimeSpan.FromSeconds(model.IdleTimeoutSeconds.Value) : null,
            MaxQueue = defaults.MaxQueue,
            PrefillChunk = defaults.PrefillChunk,
            PrefixCache = defaults.PrefixCache,
            KvStateSnapshots = defaults.KvStateSnapshots,
            StepTokens = defaults.StepTokens,
            MinStepTokens = defaults.MinStepTokens,
            StepFloor = defaults.StepFloor,
            StepRowsPerSequence = defaults.StepRowsPerSequence,
            KvAdmission = defaults.KvAdmission,
            NgramDraftTokens = defaults.NgramDraftTokens,
            MtpDraftTokens = defaults.MtpDraftTokens,
            SpeculationMaxBatch = defaults.SpeculationMaxBatch,
            Description = model.Description ?? model.Path,
        };
    }
}
