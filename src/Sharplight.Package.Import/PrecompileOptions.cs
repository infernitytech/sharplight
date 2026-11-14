using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Package.Import;

/// <summary>How <see cref="PackagePrecompiler"/> compiles a package.</summary>
public sealed class PrecompileOptions
{
    /// <summary>Default <see cref="ContextLength"/>.</summary>
    public const int DefaultContextLength = 4096;

    /// <summary>Default <see cref="BatchTokens"/>.</summary>
    public const int DefaultBatchTokens = 512;

    /// <summary>
    /// Context length language-model graphs are compiled for. Stored kernels are only used by loads with the same value, so give
    /// the context length the model will run with.
    /// </summary>
    public int ContextLength { get; init; } = DefaultContextLength;

    /// <summary>Rows per forward pass (language models) or token bound (other graphs); must match the runtime's.</summary>
    public int BatchTokens { get; init; } = DefaultBatchTokens;

    /// <summary>
    /// Sequences of the serving batch decoder to compile (as the server's <c>--slots</c>), so a served model finds its batch
    /// decoder's kernels; 1 (the default) skips it.
    /// </summary>
    public int Slots { get; init; } = 1;

    /// <summary>Paged KV cache of the serving batch decoder in tokens (as the server's <c>--kv-tokens</c>), or <see langword="null"/>.</summary>
    public long? KvCacheTokens { get; init; }

    /// <summary>
    /// Speculative decoding with this many draft tokens (as <c>--speculate-mtp</c> / <c>--speculate-ngram</c>), or 0: also
    /// compiles the decoders that verify drafts (recurrent ops in rollback form) for one sequence (the CLI) and for
    /// <see cref="Slots"/>, and the multi-token-prediction head when the model has one.
    /// </summary>
    public int DraftTokens { get; init; }

    /// <summary>Upper bounds of further symbolic dimensions of non-LLM graphs (e.g. <c>batch</c>).</summary>
    public IReadOnlyDictionary<string, long> SymbolBounds { get; init; } = new Dictionary<string, long>();

    /// <summary>
    /// Bounds a model's own metadata implies for symbols <see cref="SymbolBounds"/> does not give (e.g. an imported batch
    /// dimension), or <see langword="null"/>. Called with the package's definition for non-LLM graphs.
    /// </summary>
    public Func<ModelDefinition, IReadOnlyDictionary<string, long>>? DefaultSymbolBounds { get; init; }

    /// <summary>
    /// Device for operations the compile device cannot run in non-LLM graphs (typically the CPU, as when such models run), or
    /// <see langword="null"/>. Not owned.
    /// </summary>
    public IComputeDevice? FallbackDevice { get; init; }

    /// <summary>Artifact tiers to store (<see cref="ArtifactTiers"/>), or <see langword="null"/> for every tier the device produces.</summary>
    public IReadOnlySet<string>? Tiers { get; init; }

    /// <summary>
    /// Compile with the per-user kernel and pipeline cache instead of a fresh temporary one. Faster; the stored sets still hold only
    /// this model's kernels and pipelines.
    /// </summary>
    public bool UseUserKernelCache { get; init; }

    /// <summary>Device features never to use (see <c>DeviceFeatures</c>); the stored code is keyed by the features in use.</summary>
    public IReadOnlyCollection<string> DisabledFeatures { get; init; } = [];

    /// <summary>Receives progress messages, or <see langword="null"/>.</summary>
    public Action<string>? Log { get; init; }

    /// <summary>
    /// Receives structured progress of each graph's compilation (<see cref="LoadStage.Placing"/>, <see cref="LoadStage.Compiling"/>
    /// with kernel counts); counts restart for every graph of the package (<see cref="LoadProgress.Detail"/> names the device).
    /// </summary>
    public IProgress<LoadProgress>? Progress { get; init; }

    /// <summary>Cancels the precompilation between kernels.</summary>
    public CancellationToken CancellationToken { get; init; }
}
