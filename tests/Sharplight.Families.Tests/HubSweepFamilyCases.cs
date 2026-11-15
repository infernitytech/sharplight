using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Bring-up round 11: the Hugging Face <c>model_type</c>s the real-checkpoint sweep of round 10 found unsupported (docs/model-support.md section 10).
/// The tiny checkpoints come from <c>tools/mk_tiny_remote_code.py</c> (families whose modeling code ships with the checkpoint) and the other generators
/// in <c>tools/</c>; their references are greedy ids from transformers running that code (<c>tools/hf_greedy_reference.py</c>), and the GGUF files
/// llama.cpp's converter writes from them give the same ids on llama.cpp 8ca5a80 unless a case says otherwise. Every case skips when its files are missing.
/// </summary>
internal static class HubSweepFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    // Reference ids shared by a Hugging Face case and the GGUF converted from it. Declared before All, which reads them.
    private static readonly int[] Lfm3Ids = [18581, 64828, 105864, 29038, 123143, 9737, 24261, 112852, 67950, 125839, 57461, 22849, 35768, 109141, 2417, 91528, 70011, 18432];

    private static readonly int[] Spark25Ids = [127014, 35203, 9121, 63078, 97488, 1941, 28582, 74250, 101384, 30139, 107273, 63558, 117809, 120517, 35895, 88845,
        72935, 47223, 89644, 109280, 52132, 109143, 88939, 91766];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // LFM3 (remote code): short-conv layers, sliding-window and full attention layers on their own rotary bases, zero-tail partial rotation of the
        // full layers, per-head sigmoid output gate, sandwich norms. The 19th token is a 0.016 near-tie, so 18 are compared.
        new("tiny-random-lfm3 HF (lfm3, remote code)", "hf/tiny-random-lfm3", Prompt, "ids", Lfm3Ids, ""),
        new("tiny-random-lfm3 f32 GGUF (lfm3, llama.cpp)", "gguf/tiny-random-lfm3-hf-f32.gguf", Prompt, "ids", Lfm3Ids, ""),
        // Spark 2.5 / X2.6 (remote code written for transformers 4.x, run through tools/remote_code_compat.py): fused q_k_v_proj, per-head sigmoid
        // output gate, sliding layers on their own base, partial rotation of the full layers, GELU MLP (exact in transformers, tanh in llama.cpp; the
        // ids agree).
        new("tiny-random-spark2_5 HF (spark2_5, remote code)", "hf/tiny-random-spark2_5", Prompt, "ids", Spark25Ids, ""),
        new("tiny-random-spark2_5 f32 GGUF (spark2_5, llama.cpp)", "gguf/tiny-random-spark2_5-hf-f32.gguf", Prompt, "ids", Spark25Ids, ""),
    ];
}
