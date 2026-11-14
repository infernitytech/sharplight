using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// The generic-decoder families (model-support.md waves 0 and 2): Mixtral, SmolLM3, OLMo 1/2, GLM-4, ERNIE 4.5, Hunyuan, EXAONE 3/4, Seed-OSS, InternLM2, OpenELM, Granite and
/// Granite-MoE, MiniCPM, Phi-2, StarCoder2, Command-R and Cohere2 (GGUF only: the tiny Cohere checkpoints have a hidden size of 8, which makes their logits too close to ties to compare), Falcon, GPT-NeoX/Pythia, GPT-J, GPT-2, Nemotron, StableLM, Apertus, AFM (Arcee) and Gemma 1.
/// Hugging Face cases compare greedy token ids with transformers (fp32 compute on the checkpoint's weights); GGUF cases compare the continuation with llama.cpp b11177.
/// Q4_K_M files of small models are sensitive to the matmul summation order (llama.cpp's own CUDA and CPU builds disagree on several of them), so a few
/// cases compare only the leading tokens, or the llama.cpp CPU output, as the case name says. Tiny random checkpoints other than the ones in model-support.md section 8.2
/// (GPT-2, GPT-J, Nemotron, Gemma, Falcon) come from <c>tools/mk_tiny_decoders.py</c>. All cases skip when their files are missing.
/// </summary>
internal static class DecoderFamilyCases
{
    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Hugging Face checkpoints, token ids from transformers.
        new("tiny-random-MixtralForCausalLM HF (Mixtral, transformers)", "hf/tiny-random-MixtralForCausalLM", HfPrompt, "ids", [3974,26043,8094,19088,8364,19861,8119,10943,10918,11074,25227,8352,23413,17694,7903,31488,12458,7617,14506,28732,12012,11978,14418,11421], ""),
        new("tiny-random-smollm3 HF (smollm3, transformers)", "hf/tiny-random-smollm3", HfPrompt, "ids", [30578,56602,10574,56602,84319,56602,84319,56602,84319,56602,84319,56602,84319,56602,84319,56602,84319,56602,84319,56602,84319,56602,84319,56602], ""),
        new("tiny-random-Olmo2ForCausalLM HF (olmo2, transformers)", "hf/tiny-random-Olmo2ForCausalLM", HfPrompt, "ids", [67382,42531,89509,31067,94733,34333,89509,31067,94733,34333,89509,31067,94733,34333,89509,31067,94733,34333,89509,31067,94733,34333,89509,31067], ""),
        new("tiny-random-glm-4 HF (glm4, transformers)", "hf/tiny-random-glm-4", HfPrompt, "ids", [146453,70589,48702,70589,119316,148849,47382,145416,70589,119316,14731,5139,92049,75835,36401,70589,11252,132232,72270,131801,53985,20641,137638,107578], ""),
        new("tiny-random-ernie-4.5 HF (ernie4_5, transformers)", "hf/tiny-random-ernie-4.5", HfPrompt, "ids", [18348,91990,95854,91990,95854,91990,95854,91990,95854,91990,95854,91990,95854,91990,95854,91990,95854,18396,39574,18396,39574,18396,39574,18396], ""),
        new("tiny-random-seed-oss HF (seed_oss, transformers)", "hf/tiny-random-seed-oss", HfPrompt, "ids", [111876,111876,111876,111876,111876,111876,111876,111876,111876,111876,111876,111876,111876,111876,111876,111876,111876,111876,19722,111876,19722,111876,19722,111876], ""),
        new("ERNIE-4.5-0.3B-PT HF (ERNIE-4.5-0.3B-PT, transformers)", "hf/ERNIE-4.5-0.3B-PT", HfPrompt, "ids", [93938,6720,93938,326,18334,93937,23,23,700,9689,315,10298,357,11855,93938,724,357,274,5714,418,750,1008,274,4011], ""),
        new("Hunyuan-0.5B-Instruct HF (Hunyuan-0.5B-Instruct, transformers)", "hf/Hunyuan-0.5B-Instruct", HfPrompt, "ids", [478,478,478,478,478,478,478,478,478,478,478,478,478,478,478,478,478,478,478,478,478,478,478,478], ""),
        new("EXAONE-4.0-1.2B HF (EXAONE-4.0-1.2B, transformers)", "hf/EXAONE-4.0-1.2B", HfPrompt, "ids", [686,3993,686,3993,686,3993,686,3993,686,3993,686,3993,686,3993,686,3993,686,3993,686,3993,686,3993,686,3993], ""),
        new("granite-3.1-1b-a400m-instruct HF (granite-3.1-1b-a400m-instruct, transformers)", "hf/granite-3.1-1b-a400m-instruct", HfPrompt, "ids", [30,27668,30,461,322,5549,101,32,2030,1597,312,11297,688,1597,312,4193,432,322,43652,902,10464,461,322,43652], ""),
        new("SmolLM3-3B HF (SmolLM3-3B, transformers)", "hf/SmolLM3-3B", HfPrompt, "ids", [11,11401,11,323,36105,13,1102,574,1101,264,3363,430,574,2162,311,1690,11495,1274,11,2737,13820,11,16483,11], "", Large: true),
        new("tiny-random-granite HF (granite, transformers)", "hf/tiny-random-granite", HfPrompt, "ids", [5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549], ""),
        new("tiny-random-granite-moe HF (granitemoe, transformers)", "hf/tiny-random-granite-moe", HfPrompt, "ids", [5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549,5549], ""),
        new("tiny-random-PhiForCausalLM HF (phi, transformers)", "hf/tiny-random-PhiForCausalLM", HfPrompt, "ids", [236,352,330,492,387,774,520,136,947,79,315,359,779,421,816,852,438,175,906,876,494,172,11,566], ""),
        new("tiny-random-Starcoder2ForCausalLM HF (starcoder2, transformers)", "hf/tiny-random-Starcoder2ForCausalLM", HfPrompt, "ids", [84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84], ""),
        new("tiny-random-GPTNeoXForCausalLM HF (gpt_neox, transformers)", "hf/tiny-random-GPTNeoXForCausalLM", HfPrompt, "ids", [751,895,157,497,318,966,8,790,1017,1014,428,375,602,425,697,819,733,205,429,661,596,468,188,979], ""),
        new("tiny-random-OlmoForCausalLM HF (olmo, transformers)", "hf/tiny-random-OlmoForCausalLM", HfPrompt, "ids", [9571,26700,48280,18328,6666,46517,4724,37449,33714,37523,49992,49715,30695,37618,40384,26798,17228,21415,23774,18430,40502,20041,46155,2826], ""),
        new("tiny-random-StableLmForCausalLM HF (stablelm, transformers)", "hf/tiny-random-StableLmForCausalLM", HfPrompt, "ids", [833,388,663,804,240,882,438,214,804,240,976,652,380,419,75,362,603,513,359,928,113,865,753,485], ""),
        new("tiny-random-apertus HF (apertus, transformers)", "hf/tiny-random-apertus", HfPrompt, "ids", [102189,38378,36614,115617,12645,92943,63656,53945,106613,73344,57606,90642,2323,30545,102189,38378,36614,115617,12645,92943,63656,53945,106613,73344], ""),
        new("tiny-random-gpt2 HF (gpt2, transformers)", "hf/tiny-random-gpt2", HfPrompt, "ids", [89,389,389,389,389,389,389,389,389,389,389,368,368,389,389,926,606,368,389,389,389,926,368,368], ""),
        new("tiny-random-gptj HF (gptj, transformers)", "hf/tiny-random-gptj", HfPrompt, "ids", [314,208,523,88,973,653,287,646,646,646,646,646,646,646,646,646,646,646,646,646,646,646,646,646], ""),
        new("tiny-random-nemotron HF (nemotron, transformers)", "hf/tiny-random-nemotron", HfPrompt, "ids", [244,792,445,772,84,591,884,84,244,6,228,84,527,197,6,423,388,711,563,659,464,886,659], ""),
        new("tiny-random-gemma HF (gemma, transformers)", "hf/tiny-random-gemma", HfPrompt, "ids", [368,142,909,933,309,706,239,411,399,839,345,303,596,627,871,41,7,140,965,52,683,714,7,755], ""),
        new("tiny-random-falcon HF (falcon, transformers)", "hf/tiny-random-falcon", HfPrompt, "ids", [809,809,809,815,815,815,815,815,815,815,815,815,815,815,815,815,815,815,815,815,815,815,815,815], ""),
        // Tiny random checkpoints of the remote-code families transformers 5 cannot load; reference = llama.cpp CPU on the f32 GGUF the converter writes from the checkpoint.
        new("tiny-random-internlm2 HF (llama.cpp)", "hf/tiny-random-internlm2", HfPrompt, "text", [], "抹茶实业抹茶实业抹茶实业抹茶实业抹茶实业抹茶实业", Tokens: 12),
        new("tiny-random-minicpm4 HF (llama.cpp)", "hf/tiny-random-minicpm4", HfPrompt, "text", [], " coupons给出的选项中飱patibility飱patibility飱patibility飱patibility飱patibility", Tokens: 12),
        // GGUF files, continuation from llama.cpp.
        new("Mistral-7B-Instruct-v0.3 Q4_K_M GGUF (llama.cpp)", "gguf/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris, but the largest city is Marseille.\n\nMarseille", Tokens: 16, Large: true),
        new("SmolLM3-3B Q4_K_M GGUF (llama.cpp)", "gguf/SmolLM3-3B/SmolLM3-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris. The Eiffel Tower is a famous landmark in Paris. The E", Tokens: 16),
        new("OLMo-2-0425-1B-Instruct Q4_K_M GGUF (llama.cpp)", "gguf/OLMo-2-0425-1B-Instruct/OLMo-2-0425-1B-Instruct-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris, and the largest city in France is Lyon. Lyon is known for its", Tokens: 16),
        new("GLM-4-9B-0414 Q4_K_M GGUF (llama.cpp)", "gguf/GLM-4-9B-0414/GLM-4-9B-0414-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris. It is one of the most beautiful cities in the world. It is", Tokens: 16, Large: true),
        new("ERNIE-4.5-0.3B-PT Q4_K_M GGUF (llama.cpp, first 3 tokens)", "gguf/ERNIE-4.5-0.3B-PT/ERNIE-4.5-0.3B-PT-Q4_K_M.gguf", "The capital of France is", "text", [], "...\n\n", Tokens: 3),
        new("Hunyuan-0.5B-Instruct Q4_K_M GGUF (llama.cpp)", "gguf/Hunyuan-0.5B-Instruct/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf", "The capital of France is", "text", [], " \" \" \" \" \" \" \" \" \" \" \" \" \" \" \" \"", Tokens: 16),
        new("EXAONE-3.5-2.4B-Instruct Q4_K_M GGUF (llama.cpp)", "gguf/EXAONE-3.5-2.4B-Instruct/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris. Paris is located in northern France on the Seine River. Paris is", Tokens: 16),
        new("EXAONE-4.0-1.2B Q4_K_M GGUF (llama.cpp)", "gguf/EXAONE-4.0-1.2B/EXAONE-4.0-1.2B-Q4_K_M.gguf", "The capital of France is", "text", [], " is is is is is is is", Tokens: 16),
        new("internlm2_5-1_8b-chat Q4_K_M GGUF (llama.cpp CPU)", "gguf/internlm2_5-1_8b-chat/internlm2_5-1_8b-chat-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris. The French language is spoken in France, as well as in several other", Tokens: 16),
        new("OpenELM-270M-Instruct Q4_K_M GGUF (llama.cpp)", "gguf/OpenELM-270M-Instruct/OpenELM-270M-Instruct.Q4_K_M.gguf", "The capital of France is", "text", [], " the city of Paris, which is the largest metropolis in the country. The", Tokens: 16),
        new("granite-3.3-2b-instruct Q4_K_M GGUF (llama.cpp)", "gguf/granite-3.3-2b-instruct/granite-3.3-2b-instruct-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris.\nThe capital of Germany is Berlin.\nThe capital of", Tokens: 16),
        new("granite-3.1-1b-a400m-instruct Q4_K_M GGUF (llama.cpp CPU)", "gguf/granite-3.1-1b-a400m-instruct/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris.\n\nParis is the capital of France.\n\nPar", Tokens: 16),
        new("MiniCPM4.1-8B Q4_K_M GGUF (llama.cpp)", "gguf/MiniCPM4.1-8B/MiniCPM4.1-8B-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris.\n[A]. Berlin\n[B]. Madrid\n[", Tokens: 16, Large: true),
        new("phi-2 Q4_K_M GGUF (llama.cpp)", "gguf/phi-2/phi-2.Q4_K_M.gguf", "The capital of France is", "text", [], " Paris.", Tokens: 16),
        new("starcoder2-3b Q4_K_M GGUF (llama.cpp)", "gguf/starcoder2-3b/starcoder2-3b-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris.\",\n        \"The capital of Germany is Berlin.\",\n        \"The", Tokens: 16),
        new("aya-expanse-8b Q4_K_M GGUF (llama.cpp)", "gguf/aya-expanse-8b/aya-expanse-8b-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris. It is located in the north-central part of the country, on", Tokens: 16, Large: true),
        new("c4ai-command-r7b-12-2024 Q4_K_M GGUF (llama.cpp)", "gguf/c4ai-command-r7b-12-2024/c4ai-command-r7b-12-2024-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris, which is also the largest city in the country. The city is located", Tokens: 16, Large: true),
        new("pythia-160m Q4_K_M GGUF (llama.cpp, first 12 tokens)", "gguf/pythia-160m/pythia-160m.Q4_K_M.gguf", "The capital of France is", "text", [], " located in the city of St-Germain-en-", Tokens: 12),
        new("gpt2 Q4_K_M GGUF (llama.cpp, first 15 tokens)", "gguf/gpt2/gpt2.Q4_K_M.gguf", "The capital of France is", "text", [], " the capital of the Republic of France, and the capital of the Republic of", Tokens: 15),
        new("stablelm-2-1_6b Q4_K_M GGUF (llama.cpp, first 4 tokens)", "gguf/stablelm-2-1_6b/stablelm-2-1_6b.q4_k_m.gguf", "The capital of France is", "text", [], " a city of", Tokens: 4),
        new("Nemotron-Mini-4B-Instruct Q4_K_M GGUF (llama.cpp)", "gguf/Nemotron-Mini-4B-Instruct/Nemotron-Mini-4B-Instruct-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris, which is also the country's largest city.\n\n- The", Tokens: 16),
        new("Apertus-8B-Instruct-2509 Q4_K_M GGUF (llama.cpp, first 11 tokens)", "gguf/Apertus-8B-Instruct-2509/Apertus-8B-Instruct-2509-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris, which is also the country's largest city.", Tokens: 11, Large: true),
        new("AFM-4.5B Q4_K_M GGUF (llama.cpp)", "gguf/AFM-4.5B/AFM-4.5B-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris,\" is a true statement.\n\nTrue.\n\nThe Eiffel Tower is", Tokens: 16),
        // ALiBi (attention.alibi_max_bias): the Vulkan attention kernels (scalar, split-K decode, flash) add the score bias themselves.
        new("tiny-random-bloom HF (bloom ALiBi, transformers)", "hf/tiny-random-bloom", HfPrompt, "ids",
            [1003, 999, 779, 764, 951, 143, 951, 143, 951, 143, 951, 143, 951, 143, 951, 143, 951, 143, 951, 143, 951, 143, 951, 143], ""),
        new("tiny-random-mpt HF (mpt ALiBi, transformers)", "hf/tiny-random-mpt", HfPrompt, "ids",
            [321, 140, 433, 90, 186, 79, 231, 465, 665, 592, 476, 684, 883, 119, 214, 540, 200, 870, 681, 214, 540, 887, 816, 899], ""),
        // llama.cpp's own CPU and Vulkan builds disagree on this Q4_K_M file at the fourth token (a near-tie); the reference is its Vulkan build.
        new("bloom-560m Q4_K_M GGUF (llama.cpp Vulkan)", "gguf/bloom-560m/bloom-560m.Q4_K_M.gguf", "The capital of France is", "text", [],
            " Paris. The city is located in the south-east of France. The capital is Paris. The city", Tokens: 20),
    ];

    private const string HfPrompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";
}
