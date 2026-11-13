namespace Sharplight.Convert;

/// <summary>Parsed <c>precompile</c> options.</summary>
/// <param name="PackagePath">The package to compile and append to.</param>
/// <param name="Device">The device to compile on.</param>
/// <param name="Tiers">Artifact tiers to store (<see cref="Runtime.ArtifactTiers"/>), or null for every tier the device produces.</param>
/// <param name="ContextLength">Context length to compile LLM graphs for; must match the runtime's for the stored code to be used.</param>
/// <param name="BatchTokens">Rows per forward pass (LLMs) or token bound (other graphs); must match the runtime's.</param>
/// <param name="SymbolBounds">Bounds of further symbolic dimensions of non-LLM graphs (e.g. <c>batch</c>).</param>
/// <param name="UseUserKernelCache">Compile with the per-user kernel and pipeline cache instead of a fresh temporary one.</param>
internal sealed record PrecompileSettings(
    string PackagePath,
    DeviceSpec Device,
    IReadOnlySet<string>? Tiers,
    int ContextLength,
    int BatchTokens,
    IReadOnlyDictionary<string, long> SymbolBounds,
    bool UseUserKernelCache);
