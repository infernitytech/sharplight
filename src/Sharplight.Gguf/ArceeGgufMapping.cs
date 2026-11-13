namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>arcee</c> (AFM-4.5B): the llama graph with a plain squared-ReLU MLP, rows permuted for interleaved rope; YaRN scaling arrives as metadata or <c>rope_freqs</c>.</summary>
internal sealed class ArceeGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["arcee"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "relu2";
}
