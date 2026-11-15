namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>xing4_0</c> (Xing4.0): DeepSeek-V3's latent attention and MoE layers on manifold-constrained hyper-connections
/// (<c>hyper_connection.count</c> / <c>sinkhorn_iterations</c> / <c>epsilon</c>, tensors <c>hc_attn_*</c> / <c>hc_ffn_*</c> as in glm5-next) in
/// Xing4.0's variant: no eps on the collapse weights, mixing logits clamped to ±30 before the row softmax without an eps after it, stream k
/// mixing the old streams with row k of the matrix, and the streams averaged before the output norm. The NextN layer after the trunk has no
/// hyper-connections.
/// </summary>
internal sealed partial class LatentDecoderGgufMapping
{
    private static bool IsXing40(GgufMetadataView g) => g.GgufArchitecture == "xing4_0";

    private static void AddXing40HyperParameters(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        md[a + "hyper_connections.streams"] = g.Int("hyper_connection.count");
        md[a + "hyper_connections.eps"] = g.Float("hyper_connection.epsilon", 1e-6f);
        md[a + "hyper_connections.sinkhorn_iterations"] = g.Int("hyper_connection.sinkhorn_iterations", 20);
        md[a + "hyper_connections.pre_eps"] = 0f;
        md[a + "hyper_connections.mixing_eps"] = 0f;
        md[a + "hyper_connections.mixing_clamp"] = 30f;   // fixed in llama.cpp's graph (the reference kernel's clamp)
        md[a + "hyper_connections.mixing_transposed"] = true;
    }
}
