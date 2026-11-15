using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// The language model of DeepSeek-OCR, DeepSeek-OCR-2 and Unlimited-OCR (llama.cpp <c>deepseek2-ocr</c>): DeepSeek-V2's MoE layer with plain
/// multi-head attention instead of latent attention. The tiny float32 checkpoints come from <c>tools/mk_tiny_deepseek_ocr.py</c> (transformers
/// 5.20, which saves them with the remote-code tensor names); references are greedy ids and prompt logits from transformers
/// (<c>tools/hf_greedy_reference.py --logits</c>). The DeepSeek-OCR checkpoint holds the Unlimited-OCR weights in DeepSeek-OCR's own config
/// layout; its GGUF (llama.cpp's <c>convert_hf_to_gguf.py</c>) gives the same ids in llama.cpp 8ca5a80. Every case skips when its files are missing.
/// </summary>
internal static class DeepseekOcrFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] UnlimitedIds = [30696, 69390, 59871, 41794, 44525, 79736, 60906, 51273, 2203, 10266, 105884, 17973, 123613, 41468, 120728,
        71116, 83946, 71396, 12028, 36913, 29713, 96396, 10899, 59824];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // DeepSeek-OCR-2 (deepseek_ocr2): a dense first layer, then softmax-routed experts (8, top 3, not renormalized) plus two shared experts.
        new("tiny-random-deepseek-ocr2 HF (deepseek_ocr2, transformers)", "hf/tiny-random-deepseek-ocr2", Prompt, "ids",
            [70621, 124212, 48554, 73083, 87092, 81680, 42298, 80755, 420, 8891, 35209, 45067, 61287, 89320, 106330, 25727, 103158, 71833, 32499, 98878,
             61338, 36829, 38400, 35678], ""),
        // Unlimited-OCR (unlimited_ocr) with its reference sliding window wider than the sequence, i.e. full attention as in llama.cpp.
        new("tiny-random-unlimited-ocr HF (unlimited_ocr, transformers)", "hf/tiny-random-unlimited-ocr", Prompt, "ids", UnlimitedIds, ""),
        // The same weights as DeepSeek-OCR's remote-code checkpoint (deepseek_vl_v2, DeepseekOCRForCausalLM) and as its GGUF (deepseek2-ocr).
        new("tiny-random-deepseek-ocr HF (deepseek_vl_v2, transformers)", "hf/tiny-random-deepseek-ocr", Prompt, "ids", UnlimitedIds, ""),
        new("tiny-random-deepseek-ocr f32 GGUF (deepseek2-ocr, llama.cpp)", "gguf/tiny-random-deepseek-ocr-f32.gguf", Prompt, "ids", UnlimitedIds, ""),
    ];
}
