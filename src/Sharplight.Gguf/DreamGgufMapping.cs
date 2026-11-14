namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>dream</c> (Dream: Qwen2 layers with bidirectional attention), a masked-diffusion language model: canonical
/// <c>attention.causal</c> = false and <c>diffusion.shift_logits</c> (the file's <c>diffusion.shift_logits</c>, else true, as llama.cpp's
/// diffusion example assumes for Dream).
/// </summary>
internal sealed class DreamGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["dream"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g) => AddDiffusion(md, g, shiftByDefault: true);

    /// <summary>The bidirectional-attention and logit-shift keys of a masked-diffusion model.</summary>
    internal static void AddDiffusion(Dictionary<string, object> md, GgufMetadataView g, bool shiftByDefault)
    {
        md[GgufDecoderHyperParameters.Prefix + "attention.causal"] = false;
        md[GgufDecoderHyperParameters.Prefix + "diffusion.shift_logits"] =
            g.All.TryGetValue("diffusion.shift_logits", out var declared) ? Convert.ToBoolean(declared) : shiftByDefault;
    }
}
