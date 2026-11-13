namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>smollm3</c> (SmolLM3): the Llama decoder where <c>no_rope_layers</c> (0 = NoPE) switches the rotary embedding off per layer; optional sliding window from <c>layer_types</c>.</summary>
internal sealed class SmolLm3HfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["smollm3"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (IntArray(c, "no_rope_layers") is { } ropeLayers)
        {
            md["decoder.rope.layers"] = ropeLayers;
        }

        if (c.GetBool("use_sliding_window") == true)
        {
            AddSlidingLayers(md, c);
        }
    }
}
