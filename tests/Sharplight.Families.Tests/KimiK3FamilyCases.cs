using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Kimi K3 and K3.1 on the latent decoder: Kimi-Linear layers with block attention residuals, the situ activation, latent MoE, full-rank delta-attention gates
/// with the bounded decay and a sigmoid output gate on the latent attention; K3.1 adds a DSA indexer without rotary embedding (16 heads, top 8
/// positions, so the 36-token prompt runs in the sparse regime) to the latent-attention layers. The tiny float32 checkpoints come from <c>tools/mk_tiny_kimi_k3.py</c>
/// (the remote modeling code of the tiny-random/kimi-k3 checkpoint, run on transformers 5.20 through <c>tools/kimi_k3_reference.py</c>); the
/// Hugging Face case checks the greedy ids and prompt logits of that code.
/// The GGUF file (<c>tools/convert_tiny_gguf.py</c> with llama.cpp 8ca5a80's converter) has different semantics: the converter keeps the first
/// <c>num_heads</c> values of the per-key-channel <c>A_log</c> [head_dim] as one decay rate per head, and llama.cpp normalizes the compressed
/// query and latent with <c>rms_norm_eps</c>. Its reference ids are those of llama.cpp with the prompt decoded token by token (the C API, f32 KV
/// cache, CPU), which are also those of the remote code evaluated with these semantics (<c>tools/kimi_k3_gguf_reference.py</c>; smallest top-2
/// margin 0.005). llama.cpp's batched prefill differs from both by up to 0.05 in the logits (checked with a constant <c>A_log</c>, where the two
/// semantics coincide) and so picks another token at step 22 for K3 and at step 2 for K3.1. Every case skips when its files are missing.
/// </summary>
internal static class KimiK3FamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] KimiK3Ids = [346, 523, 71, 552, 122, 312, 12, 640, 591, 98, 655, 870, 976, 737, 851, 212, 256, 529, 403, 336, 134, 521, 1001, 541];
    private static readonly int[] KimiK31Ids = [177, 981, 219, 875, 38, 616, 596, 701, 850, 197, 261, 66, 140, 212, 568, 342, 763, 139, 947, 926, 1019, 370, 58, 779];
    private static readonly int[] KimiK31GgufIds = [653, 512, 629, 945, 474, 560, 527, 892, 876, 913, 89, 715, 349, 707, 92, 218, 656, 525, 952, 571, 306, 721, 574, 412];
    private static readonly int[] KimiK3GgufIds = [669, 479, 704, 37, 203, 715, 472, 169, 237, 280, 214, 757, 669, 779, 520, 513, 1002, 966, 1, 251, 581, 521, 921, 744];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Eight layers: delta attention except the latent-attention layers 4 and 8, attention residuals in blocks of three layers, a dense first layer.
        new("tiny-random-kimi-k3-f32 HF (kimi_k3, transformers)", "hf/tiny-random-kimi-k3-f32", Prompt, "ids", KimiK3Ids, ""),
        new("tiny-random-kimi-k3 f32 GGUF (kimi-k3, llama.cpp token by token)", "gguf/tiny-random-kimi-k3-f32.gguf", Prompt, "ids",
            KimiK3GgufIds, ""),
        // K3.1: the same layers with the indexer on layers 4 and 8.
        new("tiny-random-kimi-k31-f32 HF (kimi_k3 with dsa_config, transformers)", "hf/tiny-random-kimi-k31-f32", Prompt, "ids", KimiK31Ids, ""),
        new("tiny-random-kimi-k31 f32 GGUF (kimi-k31, llama.cpp token by token)", "gguf/tiny-random-kimi-k31-f32.gguf", Prompt, "ids",
            KimiK31GgufIds, ""),
    ];
}
