namespace Sharplight.Architectures;

/// <summary>The settings of one Dia generation.</summary>
/// <param name="Guidance">The classifier-free guidance scale (1 turns the unconditional pass off).</param>
/// <param name="Temperature">The sampling temperature (at most 0: greedy; below 1 counts as 1, as in the reference).</param>
/// <param name="TopK">Keeps this many of the highest logits when sampling (0: all).</param>
/// <param name="TopP">Keeps the smallest set of logits whose probabilities reach this mass when sampling (1: all).</param>
/// <param name="GuidanceTopK">Restricts the guided choice to the top logits of the guided scores (0: all).</param>
/// <param name="Seed">The sampler's seed, or <see langword="null"/> for a random one.</param>
/// <param name="MaxLength">The most rows of the decoder's sequence (prompt rows included), the end of the last codebook's delay included.</param>
internal sealed record DiaGenerationSettings(float Guidance, float Temperature, int TopK, float TopP, int GuidanceTopK, long? Seed, int MaxLength);
