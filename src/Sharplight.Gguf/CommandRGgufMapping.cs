namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>command-r</c> (Command-R, Aya) and <c>cohere2</c> (Command-R7B, Command-A): layer norm without bias, parallel residual around
/// one shared norm, interleaved rotary pairs (stored unpermuted), logits scaled by <c>logit_scale</c>. Cohere2 alternates sliding-window layers
/// with rotary embedding and every fourth, global layer without it.
/// </summary>
internal sealed class CommandRGgufMapping : GgufDecoderFamilyMapping
{
    private const int SlidingPattern = 4;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["command-r", "cohere2"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "residual.parallel"] = true;
        if (g.Has("logit_scale"))
        {
            md[prefix + "logit_scale"] = g.Float("logit_scale");
        }

        if (g.GgufArchitecture != "cohere2")
        {
            return;
        }

        int layerCount = g.Int("block_count");
        var sliding = GgufMappingHelpers.SlidingLayers(layerCount, SlidingPattern);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
    }
}
