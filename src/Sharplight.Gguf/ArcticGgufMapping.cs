namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>arctic</c> (Snowflake Arctic): a dense SwiGLU MLP on the attention residual plus a second, softmax-routed expert branch that
/// reads the layer input through its own norm (<c>ffn_norm_exps</c>, canonical <c>moe_norm.weight</c>); q/k stored in permuted order.
/// </summary>
internal sealed class ArcticGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal) { ["ffn_norm_exps.weight"] = "moe_norm.weight" };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["arctic"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;
}
