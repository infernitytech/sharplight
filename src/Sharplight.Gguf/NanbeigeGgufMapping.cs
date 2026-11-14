namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>nanbeige</c> (Nanbeige 4.2): a Llama block stack run <c>num_loops</c> times with shared weights (each pass with KV caches
/// of its own), the output norm applied between the passes unless <c>skip_loop_final_norm</c>; q/k stored in llama.cpp's permuted order.
/// </summary>
internal sealed class NanbeigeGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["nanbeige"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "layer_loops"] = g.Int("num_loops", 1);
        md[prefix + "loop_norm"] = !g.Bool("skip_loop_final_norm", false);
    }
}
