using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// GGUF mappings checked on tiny random f32 files, with the greedy token ids of llama.cpp b11177 (CPU, llama-server with <c>return_tokens</c>) as the reference:
/// Qwen v1, XVERSE and ChatGLM (written by <c>tools/mk_tiny_gguf.py</c>, no Hugging Face class exists for them) and the
/// dots.llm1, MiniMax-M2, Hunyuan (dense and MoE), EXAONE-MoE, Falcon, Arcee Trinity, MiMo-V2-Flash and Laguna files converted with llama.cpp's convert_hf_to_gguf.py
/// from the tiny-random checkpoints (<c>tools/mk_tiny_moe.py</c> writes the llama.cpp-compatible variants; the converter needs the unknown tiny-tokenizer hash mapped
/// (<c>get_vocab_base_pre</c> returning "gpt-2"), the dots1 and EXAONE-MoE gating function set to sigmoid, Hunyuan's rope-dimension assertion relaxed, and
/// <c>--no-mtp</c> for MiMo-V2). The Falcon file uses the <c>falcon</c> pre-tokenizer.
/// The files live in ~/models/gguf; every case skips when its file is missing.
/// </summary>
internal static class TinyGgufFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        new("tiny-random-qwen1 f32 GGUF (qwen, llama.cpp)", "gguf/tiny-random-qwen1-f32.gguf", Prompt, "ids",
            [920, 915, 819, 965, 474, 563, 586, 586, 142, 661, 104, 256, 694, 966, 966, 966, 166, 938, 27, 469, 812, 595, 297, 265], ""),
        new("tiny-random-xverse f32 GGUF (xverse, llama.cpp)", "gguf/tiny-random-xverse-f32.gguf", Prompt, "ids",
            [859, 762, 86, 849, 545, 107, 540, 646, 830, 931, 1008, 19, 859, 421, 611, 1010, 806, 523, 164, 673, 421, 779, 101, 589], ""),
        new("tiny-random-chatglm f32 GGUF (chatglm, llama.cpp)", "gguf/tiny-random-chatglm-f32.gguf", Prompt, "ids",
            [454, 984, 644, 202, 408, 958, 463, 567, 202, 935, 202, 744, 683, 551, 305, 553, 165, 532, 113, 517, 461, 780, 856, 672], ""),
        new("tiny-random-dots1-llamacpp f32 GGUF (dots1, llama.cpp)", "gguf/tiny-random-dots1-llamacpp-f32.gguf", Prompt, "ids",
            [820, 516, 621, 510, 717, 849, 856, 220, 297, 531, 824, 103, 805, 797, 856, 121, 196, 634, 411, 705, 129, 989, 949, 256], ""),
        new("tiny-random-minimax-m2 f32 GGUF (minimax-m2, llama.cpp)", "gguf/tiny-random-minimax-m2-f32.gguf", Prompt, "ids",
            [108901, 160365, 105383, 63553, 129839, 20635, 2142, 44831, 81757, 93989, 25901, 167829, 89763, 11988, 12126, 116522, 144764, 174145, 2142, 48222, 32525, 198871, 129586, 856], ""),
        // The tiny Hunyuan models have near-tied logits (llama.cpp itself picks the 4th dense token at a 0.005 margin, the 7th MoE token at 0.001): only the leading tokens are compared.
        new("tiny-random-hunyuan-dense-v1 f32 GGUF (hunyuan-dense, llama.cpp)", "gguf/tiny-random-hunyuan-dense-v1-f32.gguf", Prompt, "ids",
            [57469, 83323, 57469], ""),
        new("tiny-random-hunyuan-v1-moe-llamacpp f32 GGUF (hunyuan-moe, llama.cpp)", "gguf/tiny-random-hunyuan-v1-moe-llamacpp-f32.gguf", Prompt, "ids",
            [906, 479, 659, 442, 304, 481], ""),
        new("tiny-random-exaone-moe f32 GGUF (exaone-moe, llama.cpp)", "gguf/tiny-random-exaone-moe-f32.gguf", Prompt, "ids",
            [77, 583, 803, 940, 458, 9, 236, 236, 676, 834, 236, 989, 3, 768, 338, 656, 619, 426, 846, 521, 656, 819, 140, 820], ""),
        new("tiny-random-hy-v3 f32 GGUF (hy-v3, llama.cpp)", "gguf/tiny-random-hy-v3-f32.gguf", Prompt, "ids",
            [854, 894, 320, 513, 349, 520, 904, 932, 894, 304, 338, 195, 711, 656, 338, 803, 338, 414, 803, 1016, 101, 321, 555, 203], ""),
        new("tiny-random-afmoe f32 GGUF (afmoe, llama.cpp)", "gguf/tiny-random-afmoe-f32.gguf", Prompt, "ids",
            [620, 299, 927, 288, 634, 914, 477, 299, 348, 383, 694, 202, 908, 346, 788, 646, 46, 985, 729, 518, 547, 647, 138, 907], ""),
        new("tiny-random-falcon f32 GGUF (falcon, falcon pre-tokenizer, llama.cpp)", "gguf/tiny-random-falcon-f32.gguf", Prompt, "ids",
            [314, 975, 500, 741, 285, 500, 741, 500, 741, 285, 500, 856, 809, 975, 500, 741, 500, 741, 500, 500, 741, 56, 856, 56], ""),
    ];
}
