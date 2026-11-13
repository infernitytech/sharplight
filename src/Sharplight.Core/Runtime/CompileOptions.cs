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

    /// <summary>Features a compilation on <paramref name="device"/> may use.</summary>
    public IReadOnlySet<string> FeaturesFor(IComputeDevice device) => DeviceFeatures.Effective(device.EnabledFeatures, DisabledFeatures);

    public long Bound(string symbol) => symbol switch
    {
        Models.WellKnown.TokenSymbol => MaxTokens,
        Models.WellKnown.ContextSymbol => ContextLength,
        _ => SymbolBounds.TryGetValue(symbol, out var v) ? v : throw new InvalidOperationException($"No bound for symbol '{symbol}'."),
    };
}
