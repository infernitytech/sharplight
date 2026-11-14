namespace Sharplight.Cpu;

/// <summary>
/// The compressed KV cache input of a compressed <c>CachedAttention</c>: one row per window of <paramref name="Compression"/> positions, rows laid
/// out like the key cache (keys and, through their prefix, values).
/// </summary>
/// <param name="Cache">The compressed cache state [entries, Hkv, D].</param>
/// <param name="Compression">Positions per entry.</param>
internal sealed record CompressedKv(TRef Cache, long Compression);
