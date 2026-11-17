namespace Sharplight.VideoGeneration;

/// <summary>One prompt's text conditioning for <see cref="VideoDiffusionGenerator"/>.</summary>
/// <param name="Hidden">The per-token conditioning [L, D].</param>
/// <param name="Mask">1 for the prompt's tokens, 0 for the padding [L].</param>
/// <param name="Pooled">The pooled embedding of a second text encoder [P] (HunyuanVideo's CLIP-L), or <see langword="null"/>.</param>
internal sealed record PromptRow(float[] Hidden, float[] Mask, float[]? Pooled = null);
