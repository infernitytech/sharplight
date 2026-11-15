namespace Sharplight.ImageGeneration;

/// <summary>The text conditioning of one or more prompts, row after row.</summary>
/// <param name="Context">Per-token conditioning [rows, L, D].</param>
/// <param name="Pooled">Pooled text embeddings [rows, P] for denoisers with added conditioning, or <see langword="null"/>.</param>
internal sealed record PromptConditioning(float[] Context, float[]? Pooled)
{
    /// <summary>All-zero conditioning of the same shape (an empty negative prompt under <see cref="DiffusionConfigKeys.ZeroEmptyNegative"/>).</summary>
    public PromptConditioning Zeros() => new(new float[Context.Length], Pooled is null ? null : new float[Pooled.Length]);

    /// <summary>This conditioning's rows followed by <paramref name="other"/>'s.</summary>
    /// <param name="other">Conditioning of the same shape per row.</param>
    public PromptConditioning Append(PromptConditioning other)
        => new([.. Context, .. other.Context], Pooled is null || other.Pooled is null ? null : [.. Pooled, .. other.Pooled]);
}
