using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Families of bring-up round 7 (docs/model-support.md). Tiny random Hugging Face checkpoints come from the generators in <c>tools/</c>, their
/// references are greedy ids and prompt logits from transformers (<c>tools/hf_greedy_reference.py --logits</c>); GGUF files are converted from
/// them with llama.cpp's converter (<c>tools/convert_tiny_gguf.py</c>) and checked against llama.cpp. Every case skips when its files are missing.
/// </summary>
internal static class ExtendedFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] ChameleonMhaIds = [8348, 8393, 8265, 8232, 8318, 8335, 8277, 8337, 8235, 8291, 8232, 8218, 1, 8335, 8292, 8334, 8232, 8311, 8296, 8437, 8209, 8336, 8199, 8348];
    // The 23rd step of the reference is a near tie (logit margin 0.002), so the cases stop before it.
    private static readonly int[] MiniMaxM3Ids = [613, 805, 449, 864, 7, 475, 81, 300, 81, 190, 190, 118, 825, 219, 616, 808, 931, 825, 355, 160, 413, 955];
    private static readonly int[] Glm5NextIds = [875, 197, 435, 728, 218, 816, 832, 662, 999, 138, 781, 950, 950, 955, 270, 731, 88, 428, 187, 167, 664, 344, 174, 13];
    private static readonly int[] ChameleonSwinIds = [8348, 8348, 8348, 8218, 8235, 8337, 8235, 8337, 8232, 8232, 8232, 8348, 8232, 8348, 8421, 8348, 8232, 8348, 8232, 8348, 8232, 8348, 8232, 8232];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Chameleon (tools/mk_tiny_chameleon.py): q/k layer norms with a gain and bias per head, the image-token logits [4, 8196) suppressed.
        new("tiny-random-chameleon HF (chameleon, per-head q/k layer norms, transformers)", "hf/tiny-random-chameleon", Prompt, "ids",
            [8350, 8430, 8334, 8346, 8394, 8360, 8374, 8261, 8358, 8374, 8339, 8402, 8220, 8199, 8402, 8393, 8246, 8410, 8359, 8365, 8394, 8345, 8422, 8306], ""),
        new("tiny-random-chameleon-mha HF (chameleon, shared q/k norms, transformers)", "hf/tiny-random-chameleon-mha", Prompt, "ids", ChameleonMhaIds, ""),
        new("tiny-random-chameleon-mha f32 GGUF (chameleon, llama.cpp)", "gguf/tiny-random-chameleon-mha-f32.gguf", Prompt, "ids", ChameleonMhaIds, ""),
        new("tiny-random-chameleon-swin HF (chameleon swin norm, transformers)", "hf/tiny-random-chameleon-swin", Prompt, "ids", ChameleonSwinIds, ""),
        // Checked against transformers: llama.cpp b11177 (and 8ca5a80) feeds the swin-norm MLP the normed attention output instead of the residual
        // stream (its MLP input is the attention branch, not attention + residual), so it continues differently from the first token.
        new("tiny-random-chameleon-swin f32 GGUF (chameleon swin norm, transformers ids)", "gguf/tiny-random-chameleon-swin-f32.gguf", Prompt, "ids", ChameleonSwinIds, ""),
        // MiniMax-M3 (tools/mk_tiny_minimax_m3.py): Gemma norms, per-head q/k norms, partial rope, dense + MoE layers with SwiGLU-OAI, sparse
        // attention in its dense regime. llama.cpp 8ca5a80 continues identically.
        new("tiny-random-minimax-m3 HF (minimax_m3_vl, transformers)", "hf/tiny-random-minimax-m3", Prompt, "ids", MiniMaxM3Ids, ""),
        new("tiny-random-minimax-m3 f32 GGUF (minimax-m3, llama.cpp)", "gguf/tiny-random-minimax-m3-f32.gguf", Prompt, "ids", MiniMaxM3Ids, ""),
        // GLM-5.3-Flash (tools/mk_tiny_glm5_next.py): delta attention with the bounded gate, no-rope latent attention with a shared k-pool indexer
        // selection in its dense regime, clamped SwiGLU MLPs, hyper-connections averaged at the end. llama.cpp 8ca5a80 continues identically.
        new("tiny-random-glm5-next HF (glm5_next, transformers)", "hf/tiny-random-glm5-next", Prompt, "ids", Glm5NextIds, ""),
        new("tiny-random-glm5-next f32 GGUF (glm5-next, llama.cpp)", "gguf/tiny-random-glm5-next-f32.gguf", Prompt, "ids", Glm5NextIds, ""),
    ];
}
