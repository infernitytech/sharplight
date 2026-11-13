namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>ernie4_5-moe</c> (ERNIE-4.5-21B-A3B, 300B-A47B): the dense ERNIE 4.5 layer (interleaved rotary pairs, stored unpermuted)
/// with softmax-scored experts, a selection bias, renormalized weights and shared experts after the leading dense layers.
/// </summary>
internal sealed class Ernie45MoeGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["ernie4_5-moe"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";
}
