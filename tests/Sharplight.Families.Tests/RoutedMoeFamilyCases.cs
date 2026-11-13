using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// The routed-MoE families and gpt-oss / BitNet (model-support.md waves 3 and 6): GLM-4.5 MoE, ERNIE 4.5 MoE, Ling 2.0 (bailing),
/// MiniMax-M2, Hunyuan MoE, Llama 4, gpt-oss and BitNet, plus the GGUF models of SmallThinker, ERNIE-4.5-21B-A3B, gpt-oss-20b,
/// Ministral 3 and BitNet b1.58. Hugging Face cases compare greedy token ids with transformers (fp32 compute on the checkpoint's
/// weights); GGUF cases compare the continuation with llama.cpp b11177. All cases skip when their files are missing.
/// </summary>
internal static class RoutedMoeFamilyCases
{
    private const string HfPrompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private const string LongPrompt = HfPrompt + ". " + HfPrompt + ". " + HfPrompt + ". " + HfPrompt + ". " + HfPrompt + ". " + HfPrompt + ".";

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Hugging Face checkpoints, token ids from transformers.
        new("tiny-random-glm-4-moe HF (glm4_moe, transformers)", "hf/tiny-random-glm-4-moe", HfPrompt, "ids", [16511,57236,56451,19031,19031,19031,19031,19031,19031,19031,19031,19031], ""),
        new("tiny-random-ernie-4.5-moe HF (ernie4_5_moe, transformers)", "hf/tiny-random-ernie-4.5-moe", HfPrompt, "ids", [19041,102550,47987,102550,47987,102550,47987,102550,47987,102550,47987,102550], ""),
        new("tiny-random-minimax-m2 HF (minimax_m2, transformers)", "hf/tiny-random-minimax-m2", HfPrompt, "ids", [108901,160365,105383,63553,129839,20635,2142,44831], ""),
        new("tiny-random-gpt-oss HF (gpt_oss, transformers)", "hf/tiny-random-gpt-oss", HfPrompt, "ids", [115154,69901,112815,112815,112815,112815,112815,112815,112815,112815,112815,112815], ""),
        // MXFP4 experts: transformers' own dequantization (convert_moe_packed_tensors) of the same checkpoint as the reference.
        new("tiny-random-gpt-oss-mxfp4 HF (gpt_oss MXFP4, transformers)", "hf/tiny-random-gpt-oss-mxfp4", HfPrompt, "ids", [123947,192778,199875,4636,147934,4636,147934,4636,147934,4636,147934,4636], ""),
        new("tiny-random-llama-4 HF (llama4, transformers)", "hf/tiny-random-llama-4", HfPrompt, "ids", [108390,170890,17093,13840,194099,131844,130742,173695,89971,110708,165492,79380], ""),
        // Six prompts in a row (about 160 tokens) cross the 128-token attention chunk of the rotary layers.
        new("tiny-random-llama-4 HF long prompt (llama4 chunked attention, transformers)", "hf/tiny-random-llama-4", LongPrompt, "ids", [169106,46193,153014,127034,184942,177199,146815,95194,125259,18884,193513,118435], ""),
        // Tiny checkpoints from tools/mk_tiny_moe.py.
        new("tiny-random-hunyuan-v1-moe HF (hunyuan_v1_moe, transformers)", "hf/tiny-random-hunyuan-v1-moe", HfPrompt, "ids", [906,479,659,442,304,481,929,839,151,414,296,434,19,105,185,647], ""),
        new("tiny-random-dots1 HF (dots1 group-limited routing, transformers)", "hf/tiny-random-dots1", HfPrompt, "ids", [557,683,396,750,619,279,523,503,995,662,487,926,166,964,821,637], ""),
        new("tiny-random-exaone-moe HF (exaone_moe, transformers)", "hf/tiny-random-exaone-moe", HfPrompt, "ids", [77,583,803,940,458,9,236,236,676,834,236,989,3,768,338,656], ""),
        new("tiny-random-bitnet HF (bitnet, transformers)", "hf/tiny-random-bitnet", HfPrompt, "ids", [508,432,843,582,109,60,109,130,15,102,655,601,453,281,474,1009], ""),
        new("tiny-random-ministral3 HF (ministral3 yarn + query temperature, transformers)", "hf/tiny-random-ministral3", HfPrompt, "ids", [288,581,321,444,914,110,447,530,221,845,262,243,941,602,526,790], ""),
    ];
}
