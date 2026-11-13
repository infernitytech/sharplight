namespace Sharplight.Tensors;

/// <summary>What a <see cref="SymbolicFactor"/> computes.</summary>
internal enum SymbolicFactorKind
{
    /// <summary>An atomic symbol bound by the caller or by a data-dependent op.</summary>
    Symbol,
    /// <summary><c>floor(e / c)</c>.</summary>
    FloorDiv,
    /// <summary><c>ceil(e / c)</c>.</summary>
    CeilDiv,
    /// <summary><c>min(a, b)</c>.</summary>
    Min,
    /// <summary><c>max(a, b)</c>.</summary>
    Max,
}
