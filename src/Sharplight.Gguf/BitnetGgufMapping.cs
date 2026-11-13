namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>bitnet</c> (BitNet b1.58, e.g. bitnet-b1.58-2B-4T): the llama layer with ternary (TQ1_0/TQ2_0) projections, RMS
/// sub-norms in front of the attention output and MLP down projections, and a squared-ReLU gated MLP; the output head is tied to the
/// token embedding.
/// </summary>
/// <remarks>
/// The activation follows the model (Hugging Face <c>BitNetForCausalLM</c>: <c>hidden_act</c> relu2). llama.cpp b11177 builds this
/// architecture with SiLU: with SiLU, Sharplight reproduces llama.cpp's Vulkan and CPU greedy text for bitnet-b1.58-2B-4T exactly
/// (" the capital of France is the capital of France is ..."), with ReLU² it continues " Paris. Paris is a city in the north of France".
/// </remarks>
internal sealed class BitnetGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["bitnet"];

    /// <inheritdoc/>
    protected override string Activation => "relu2";
}
