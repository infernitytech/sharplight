namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>minicpm</c> (MiniCPM 2B, MiniCPM4/4.1): the Granite graph with muP multipliers; older files lack the scale keys and
/// get llama.cpp's defaults (embedding 12, residual 1.4/sqrt(layers), logit divisor 256/hidden). Longrope factors arrive as tensors.
/// </summary>
internal sealed class MiniCpmGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["minicpm"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        GgufMupScales.Add(md, g,
            embeddingScale: 12f,
            residualScale: 1.4f / MathF.Sqrt(g.Int("block_count")),
            logitDivisor: 256f / g.Int("embedding_length"));
    }
}
