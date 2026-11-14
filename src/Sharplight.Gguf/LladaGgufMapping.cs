namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>llada</c> (LLaDA: Llama layers with bidirectional attention, q/k rows permuted by the converter), a masked-diffusion language
/// model whose logits are not shifted (see <see cref="DreamGgufMapping.AddDiffusion"/>).
/// </summary>
internal sealed class LladaGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["llada"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g) => DreamGgufMapping.AddDiffusion(md, g, shiftByDefault: false);
}
