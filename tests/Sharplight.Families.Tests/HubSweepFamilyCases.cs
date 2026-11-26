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

    private static readonly int[] AionIds = [138652, 139819, 160182, 160182, 104038, 167799, 37332, 143048, 145401, 25763, 59631, 59631, 59631, 151599, 45870,
        138888, 50151];

    private static readonly int[] CloverIds = [220936, 185169, 118692, 98572, 181106, 175839, 164194, 181971, 210750, 103853, 102860, 143626, 167167,
        147275, 52747, 168635, 43474, 197560, 233695, 110073, 195413, 86318, 53563, 147735];

    private static readonly int[] PaliGemmaIds = [170034, 15765, 123992, 11679, 165971, 215518, 81916, 257132, 212102, 168353, 205411, 205059, 253655, 205059,
        235245, 11245, 42745, 210778, 106355, 98412, 33390, 152446, 51760, 69309];

    private static readonly int[] PaliGemma2Ids = [236068, 198295, 167268, 198035, 97262, 233403, 199492, 199492, 181269];

    private static readonly int[] MiMoV2Ids = [37031, 50499, 35575, 45162, 101522, 24090, 125646, 117951, 40716, 25115, 78758, 104664, 135610, 125646, 124784,
        110971, 23312, 23312, 25940, 23312, 79582, 120996, 130157, 30223];

    private static readonly int[] LlavaOneVision2Ids = [27599, 87951, 42657, 13044, 8383, 77725, 31482, 58698, 28707, 22781, 28946, 94058, 98474, 135409, 13537,
        52113, 43821, 94621, 56417, 23706, 64588, 106455, 103300];

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
        // Nemotron 3.5 Super-VL / Bolt (nemotron_h_omni, tools/mk_tiny_nemotron_h_omni.py): the tiny latent Nemotron-H below language_model.
        // with its llm_config, vision stand-ins skipped; the Puzzle form (Bolt: block_configs with per-layer expert and selection counts) has the
        // layers of the puzzle GGUF and continues as llama.cpp does on it.
        new("tiny-random-nemotron-h-omni HF (nemotron_h_omni)", "hf/tiny-random-nemotron-h-omni", Prompt, "ids", NemotronLatentIds, "", NeedsCpuFallback: true),
        new("tiny-random-nemotron-h-omni-puzzle HF (nemotron_h_omni, Puzzle block_configs)", "hf/tiny-random-nemotron-h-omni-puzzle", Prompt, "text", [],
            "emblance 얼른 propuls shortestombo�ှိ� 시장odu Dio pojedaccepted últim+(-toggle힌 Moyen", NeedsCpuFallback: true, Tokens: 16),
        // PaliGemma / PaliGemma 2 (tools/mk_tiny_paligemma.py): the Gemma / Gemma 2 text decoder below language_model., head_dim left at the Gemma
        // default (not hidden / heads in PaliGemma 2), decoded causally. The reference is transformers' eager path (HF_ATTN_IMPL=eager): its sdpa
        // path attends bidirectionally over a prompt without token_type_ids. The 10th PaliGemma 2 token is a 0.010 near-tie, so 9 are compared.
        // The tiny models scale their final norm far above one (logits up to 16), so the KV caches are kept in f32: f16 rounding of the cached
        // keys and values alone moves the CPU logits by 0.09.
        // llama.cpp has no paligemma text model.
        new("tiny-random-paligemma HF (paligemma, Gemma text)", "hf/tiny-random-paligemma", Prompt, "ids", PaliGemmaIds, "", F32KvCache: true),
        new("tiny-random-paligemma2 HF (paligemma, Gemma 2 text)", "hf/tiny-random-paligemma2", Prompt, "ids", PaliGemma2Ids, "", F32KvCache: true),
        // MiMo-V2.6-Pro (mimo_v2, remote code): the MiMo-V2-Flash layers with fused qkv_proj and swa_num_key_value_heads.
        new("tiny-random-mimo-v2 HF (mimo_v2, remote code)", "hf/tiny-random-mimo-v2", Prompt, "ids", MiMoV2Ids, ""),
        new("tiny-random-mimo-v2 f32 GGUF (mimo_v2, llama.cpp)", "gguf/tiny-random-mimo-v2-hf-f32.gguf", Prompt, "ids", MiMoV2Ids, ""),
        // Mage-VL (llava_onevision2, remote code): a Qwen3 text decoder on 1D positions; the vision tower is skipped. The 24th token is a 0.008
        // near-tie, so 23 are compared.
        new("tiny-random-llava-onevision2 HF (llava_onevision2, remote code)", "hf/tiny-random-llava-onevision2", Prompt, "ids", LlavaOneVision2Ids, ""),
        new("tiny-random-llava-onevision2 f32 GGUF (llava_onevision2, llama.cpp)", "gguf/tiny-random-llava-onevision2-hf-f32.gguf", Prompt, "ids",
            LlavaOneVision2Ids, ""),
        // Aion 1.0 (aion_instruct, remote code): Qwen3 layers with sliding layers from layer_types, YaRN on the full layers only. The 18th token is a
        // 0.012 near-tie, so 17 are compared.
        new("tiny-random-aion-instruct HF (aion_instruct, remote code)", "hf/tiny-random-aion-instruct", Prompt, "ids", AionIds, ""),
        // Clover (remote code): Qwen3.5-MoE layers decoded causally with plain rope; the block-diffusion preview tensors are left out.
        new("tiny-random-clover-moe HF (clover_moe, remote code)", "hf/tiny-random-clover-moe", Prompt, "ids", CloverIds, ""),
        new("tiny-random-clover-moe f32 GGUF (clover_moe, llama.cpp)", "gguf/tiny-random-clover-moe-hf-f32.gguf", Prompt, "ids", CloverIds, ""),

        // Real checkpoints (bf16, skipped unless downloaded). Greedy ids from transformers in f32;
        // Mage-VL's model.* names loaded as the model.language_model.* its remote code builds under transformers 5.
        new("LFM3-1.7B HF (lfm3, transformers)", "hf/LFM3-1.7B", Prompt, "ids", [20, 12187, 20, 309, 26883, 22, 810, 355, 267, 3270, 395, 355, 2037, 374, 851, 2628],
            ""),
        new("Mage-VL-Cowork HF (llava_onevision2, transformers)", "hf/Mage-VL-Cowork", Prompt, "ids",
            [11, 7674, 11, 323, 3840, 13, 576, 6722, 3283, 374, 1083, 279, 1992, 1380, 279, 3946], ""),
        // Aion 1.0 Instruct 2.3B (Microsoft): the 18th token is a 0.012 near-tie, so 17 are compared.
        new("Aion-1.0-Instruct-2.3B HF (aion_instruct, transformers)", "hf/Aion-1.0-Instruct-2.3B", Prompt, "ids",
            [11, 12808, 11, 326, 27660, 13, 1225, 382, 1217, 5542, 395, 1617, 5678, 326, 9674, 13, 623], ""),
        new("Spark-X2.6-4B HF (spark2_5, transformers)", "hf/Spark-X2.6-4B", Prompt, "ids",
            [25, 16239, 25, 317, 28548, 27, 614, 11862, 314, 17349, 390, 19500, 27, 198, 63842, 3445], ""),
    ];
}
