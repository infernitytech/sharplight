namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>baichuan</c> (Baichuan 1/2): the Llama layer (q/k rows permuted by the converter). llama.cpp tells the variants apart by depth: the 32-layer
/// 7B models are rotary, the 40-layer 13B models use ALiBi (maximum bias 8), any other depth gets no position signal. Baichuan 2's normalized output head
/// is not applied by the converter, so neither llama.cpp nor this mapping reproduces it.
/// </summary>
internal sealed class BaichuanGgufMapping : GgufDecoderFamilyMapping
{
    private const int RotaryLayerCount = 32;
    private const int AlibiLayerCount = 40;
    private const float AlibiMaxBias = 8f;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["baichuan"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        int layers = TrunkLayerCount(g);
        if (layers == RotaryLayerCount)
        {
            return;
        }

        md[GgufDecoderHyperParameters.Prefix + "rope.layers"] = new int[layers];
        if (layers == AlibiLayerCount)
        {
            md[GgufDecoderHyperParameters.Prefix + "attention.alibi_max_bias"] = AlibiMaxBias;
        }
    }
}
