namespace Sharplight.Runtime;

/// <summary>Bounds for compilation: buffers are sized for the maximum and kernels specialized.</summary>
public sealed class CompileOptions
{
    /// <summary>Maximum tokens per execution (bound for the symbolic token dimension <c>T</c>).</summary>
    public long MaxTokens { get; init; } = 512;
    /// <summary>Context length (bound for the <c>ctx</c> symbol of caches).</summary>
    public long ContextLength { get; init; } = 4096;
    /// <summary>Additional symbol bounds.</summary>
    public IReadOnlyDictionary<string, long> SymbolBounds { get; init; } = new Dictionary<string, long>();
    /// <summary>
    /// Device features (<see cref="DeviceFeatures"/>) this compilation must not use, on top of the
    /// device-level disables — e.g. to A/B test or work around a driver bug for one model.
    /// </summary>
    public IReadOnlySet<string> DisabledFeatures { get; init; } = new HashSet<string>();

    /// <summary>
    /// Allow kernels to compute f32-weight ops at reduced precision (e.g. f16 tensor-core matmuls with
    /// f32 accumulation). Ops on f16/bf16/quantized weights may always use such paths, since their
    /// error is below the weights' own precision.
    /// </summary>
    public bool AllowReducedPrecision { get; init; }

    /// <summary>
    /// Compilation cache to use on a device (kernel binaries, pipeline caches): consulted before compiling and
    /// filled with everything the compilation used. Null = the provider's own per-user caches only.
    /// </summary>
    public Func<IComputeDevice, ICompilationCache?>? CacheFor { get; init; }

    /// <summary>
    /// Receives progress of the compilation: placement (<see cref="LoadStage.Placing"/>), kernel compilation and pipeline
    /// creation or op lowering (<see cref="LoadStage.Compiling"/>, with cache hits) and weight upload
    /// (<see cref="LoadStage.Uploading"/>), counted over all segments. Reports may come from worker threads. Null = no reports.
    /// </summary>
    public IProgress<LoadProgress>? Progress { get; init; }

    /// <summary>
    /// Cancels the compilation between segments, kernels and weight uploads; a cancelled compilation throws
    /// <see cref="OperationCanceledException"/> after releasing what it allocated.
    /// </summary>
    public CancellationToken CancellationToken { get; init; }

    /// <summary>
    /// Weight buffers shared with other compilations of the same model source (null: this compilation uploads its own copy of
    /// every weight). Devices that support it (<see cref="IComputeDevice.SharesWeights"/>) reuse buffers another compiled model
    /// on the same device already holds, e.g. a draft head reusing the decoder's embedding and LM head.
    /// </summary>
    public SharedWeights? SharedWeights { get; init; }

    /// <summary>Features a compilation on <paramref name="device"/> may use.</summary>
    public IReadOnlySet<string> FeaturesFor(IComputeDevice device) => DeviceFeatures.Effective(device.EnabledFeatures, DisabledFeatures);

    /// <summary>
    /// Upper bound of a symbolic dimension: <see cref="MaxTokens"/> for the token symbol, <see cref="ContextLength"/> for the context symbol, otherwise from
    /// <see cref="SymbolBounds"/>. An expression (<c>12*N</c>) is bounded through the bounds of its symbols (<see cref="Tensors.SymbolicExpression.UpperBound"/>).
    /// </summary>
    /// <param name="symbol">Symbol name or expression text.</param>
    /// <exception cref="InvalidOperationException">No bound is known for a symbol.</exception>
    public long Bound(string symbol) => symbol switch
    {
        Models.WellKnown.TokenSymbol => MaxTokens,
        Models.WellKnown.ContextSymbol => ContextLength,
        _ when SymbolBounds.TryGetValue(symbol, out var v) => v,
        _ when !Tensors.SymbolicExpression.IsSymbolName(symbol) => Tensors.SymbolicExpression.Parse(symbol).UpperBound(Bound),
        _ => throw new InvalidOperationException($"No bound for symbol '{symbol}'."),
    };

    /// <summary>Whether <see cref="Bound"/> knows a bound for the symbol or for every symbol of the expression.</summary>
    public bool HasBound(string symbol) => symbol is Models.WellKnown.TokenSymbol or Models.WellKnown.ContextSymbol || SymbolBounds.ContainsKey(symbol)
        || (!Tensors.SymbolicExpression.IsSymbolName(symbol) && Tensors.SymbolicExpression.Parse(symbol).Atoms().All(HasBound));

    /// <summary>A copy with additional symbol bounds (existing bounds of the same symbols are replaced).</summary>
    internal CompileOptions WithSymbolBounds(IEnumerable<KeyValuePair<string, long>> bounds)
    {
        var merged = new Dictionary<string, long>(SymbolBounds, StringComparer.Ordinal);
        foreach (var (k, v) in bounds)
        {
            merged[k] = v;
        }

        return Copy(merged, Progress);
    }

    /// <summary>A copy reporting to <paramref name="progress"/>.</summary>
    internal CompileOptions WithProgress(IProgress<LoadProgress>? progress) => Copy(SymbolBounds, progress);

    private CompileOptions Copy(IReadOnlyDictionary<string, long> symbolBounds, IProgress<LoadProgress>? progress) => new()
    {
        MaxTokens = MaxTokens,
        ContextLength = ContextLength,
        SymbolBounds = symbolBounds,
        DisabledFeatures = DisabledFeatures,
        AllowReducedPrecision = AllowReducedPrecision,
        CacheFor = CacheFor,
        Progress = progress,
        CancellationToken = CancellationToken,
        SharedWeights = SharedWeights,
    };
}
