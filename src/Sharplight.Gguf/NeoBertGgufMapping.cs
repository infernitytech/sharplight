namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>neo-bert</c> (NeoBERT): a bidirectional pre-norm encoder with RMS norms, a fused q/k/v projection, an interleaved ("gptj") rotary
/// embedding and a fused SwiGLU up projection (gate rows first), finished by <c>enc.output_norm</c>. The optional classifier tensors are kept
/// under <c>classifier.*</c> and not used for embeddings.
/// </summary>
internal sealed class NeoBertGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["enc.output_norm.weight"] = "output_norm.weight",
        ["cls.weight"] = "classifier.pre.weight",
        ["cls.bias"] = "classifier.pre.bias",
        ["cls.output.weight"] = "classifier.out.weight",
        ["cls.output.bias"] = "classifier.out.bias",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["neo-bert"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? GlobalTensorOverrides => s_global;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
        => md[GgufDecoderHyperParameters.Prefix + "attention.causal"] = false;
}
