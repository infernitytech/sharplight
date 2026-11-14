using Sharplight.Runtime;

namespace Sharplight.Convert;

/// <summary>Parsed <c>precompile</c> options.</summary>
/// <param name="PackagePath">The package to compile and append to.</param>
/// <param name="Device">The live device to compile on, or null when compiling for <paramref name="Profile"/>.</param>
/// <param name="Profile">The device profile to compile for without the device, or null when compiling on <paramref name="Device"/>.</param>
/// <param name="Tiers">Artifact tiers to store (<see cref="ArtifactTiers"/>), or null for every tier the device produces.</param>
/// <param name="DisabledFeatures">Device features not to use, as when the model runs with <c>--disable-features</c>.</param>
/// <param name="ContextLength">Context length to compile LLM graphs for; must match the runtime's for the stored code to be used.</param>
/// <param name="BatchTokens">Rows per forward pass (LLMs) or token bound (other graphs); must match the runtime's.</param>
/// <param name="Slots">Sequences of the serving batch decoder to compile (as <c>sharplight-server --slots</c>); 1 or less skips it.</param>
/// <param name="KvCacheTokens">Paged KV cache of the serving batch decoder (as <c>sharplight-server --kv-tokens</c>), or null.</param>
/// <param name="DraftTokens">Draft tokens of speculative decoding to compile the verifying decoders and draft head for (0: none).</param>
/// <param name="SymbolBounds">Bounds of further symbolic dimensions of non-LLM graphs (e.g. <c>batch</c>).</param>
/// <param name="UseUserKernelCache">Compile with the per-user kernel and pipeline cache instead of a fresh temporary one.</param>
internal sealed record PrecompileSettings(
    string PackagePath,
    DeviceSpec? Device,
    DeviceProfile? Profile,
    IReadOnlySet<string>? Tiers,
    IReadOnlyList<string> DisabledFeatures,
    int ContextLength,
    int BatchTokens,
    int Slots,
    long? KvCacheTokens,
    int DraftTokens,
    IReadOnlyDictionary<string, long> SymbolBounds,
    bool UseUserKernelCache);
