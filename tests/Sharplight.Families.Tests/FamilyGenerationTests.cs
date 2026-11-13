using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Greedy generation of the newer model families against golden outputs captured from the reference implementations:
/// Hugging Face transformers (fp32 compute on the checkpoint's bf16 weights) for HF checkpoints, llama.cpp b11177 for GGUF files.
/// Every case skips when its model files are absent (under ~/models, see the case list); each runs on the CPU provider and on the
/// first Vulkan device when one is present (the discrete GPU unless SHARPLIGHT_TEST_VK_DEVICE selects another).
/// </summary>
public class FamilyGenerationTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    static FamilyGenerationTests()
    {
        GgufImporter.Register();
        HuggingFaceImporter.Register();
    }

    /// <summary>
    /// One model to generate with. <c>Kind</c> "ids": <c>Ids</c> are the expected generated token ids (transformers); "text": <c>Text</c> is the
    /// expected continuation of <c>Tokens</c> greedy tokens (llama.cpp).
    /// </summary>
    public sealed record Case(string Name, string Path, string Prompt, string Kind, int[] Ids, string Text, bool NeedsCpuFallback = false, bool Large = false, int Tokens = 40);

    private static readonly string P1 = "Once upon a time, in a small village by the sea, there lived";
    private static readonly string P2 = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";
    private const string Gemma4Story = " a young girl named Lily. Lily had hair the color of spun gold and eyes the deep blue of the ocean. She loved spending her days by the sparkling sea, collecting seashells and watching the waves dance";

    public static IEnumerable<TheoryDataRow<string>> CaseNames() => Cases.Select(c => new TheoryDataRow<string>(c.Name));

    private static readonly Case[] Cases =
    [
        // Gemma 4 E2B: per-layer embeddings, KV sharing, 256/512 head sizes, proportional rope, v-norm, layer scalars, 30-capped logits.
        new("gemma-4-E2B-it HF (transformers)", "hf/gemma-4-E2B-it", P1, "ids",
            [496, 3184, 3953, 7489, 60708, 236761, 60708, 1053, 5324, 506, 2258, 529, 69597, 5122, 532, 6114, 506, 5268, 3730, 529, 506, 12461, 236761, 2625, 9312, 11704, 1116, 2668, 684, 506, 52160, 5442, 236764, 24595, 228535, 532, 10685, 506, 11707, 12847], ""),
        new("gemma-4-E2B-it BF16 GGUF (llama.cpp)", "gguf/gemma-4-E2B-it/gemma-4-E2B-it-BF16.gguf", P1, "text", [], Gemma4Story),
        new("gemma-4-E2B-it Q8_0 GGUF (llama.cpp CPU)", "gguf/gemma-4-E2B-it/gemma-4-E2B-it-Q8_0.gguf", P1, "text", [], Gemma4Story),
        // Nemotron-H Nano 4B: Mamba-2 / attention / relu² MLP hybrid.
        new("NVIDIA-Nemotron-3-Nano-4B HF (transformers)", "hf/NVIDIA-Nemotron-3-Nano-4B", "The capital of France is", "ids",
            [6993, 2613, 5646, 1681, 2925, 1338, 3890, 1395, 2156, 2258, 31544, 1063, 1531, 4098, 8223, 1429, 11018, 1307, 1278, 3629, 21166, 1395, 2925, 10555, 1321, 2342, 1925, 1395, 2925, 1338, 3890, 2878], ""),
        new("NVIDIA-Nemotron-3-Nano-4B Q8_0 GGUF (llama.cpp)", "gguf/NVIDIA-Nemotron-3-Nano-4B/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf", "The capital of France is", "text", [],
            " Paris.\"] output: ```python\ndef extract_capital(sentence):\n    # Split the sentence into words\n    words = sentence.split()\n    #", Tokens: 32),
        // Nemotron-H with every layer kind (Mamba-2 with 2 groups, MoE with biased sigmoid routing + shared expert, attention, MLP): tiny random HF model (embedding tensor renamed to backbone.embeddings.weight).
        new("tiny-random-nemotron-h HF (transformers)", "hf/tiny-random-nemotron-h", "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art", "ids",
            [24000,92554,26777,83490,2319,37593,69072,35539,21983,120599,93,62682,121133,65255,47383,128871], "", NeedsCpuFallback: true),
        // Gemma 4 E4B finetune (NVIDIA Nemotron 3.5 Bolt E4B, arch gemma4): IQ4_NL per-layer embedding table, 42 layers with 18 shared-KV layers.
        new("NVIDIA-Nemotron-3.5-Bolt-E4B UD-Q4_K_XL GGUF (llama.cpp)", "gguf/NVIDIA-Nemotron-3.5-Bolt-E4B/NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf", "def fibonacci(n):", "text", [],
            "\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\n# Test cases\nprint"),
        // Muse Glimmer: sandwich norms, sigmoid-gated attention, NoPE global layers, weight-less embedding norm, scaled+capped logits (tiny random HF model).
        new("tiny-random-muse-glimmer HF (transformers)", "hf/tiny-random-muse-glimmer", "The capital of France is", "ids",
            [88179, 31190, 88859, 146248, 146248, 98533, 196087, 39796, 187054, 134096, 16817, 4968, 3047, 169161, 51166, 87988], ""),
        new("Muse-Glimmer-30B Q4_K_XL GGUF (llama.cpp)", "gguf/Muse-Glimmer-30B/Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "The planets of the solar system in order from the Sun are Mercury, Venus, Earth, Mars,", "text", [],
            " Jupiter, Saturn, Uranus, and Neptune. The dwarf planets are Ceres, Pluto, Haumea, Makemake, and Eris.\n\nThe planets of the solar system are Mercury,", Large: true),
        // Flint: attention sinks, sigmoid-routed experts + shared expert, NoPE full layers (tiny random model from tools/mk_tiny_flint.py; the MoE runs on the CPU when the GPU lacks MatMulId).
        new("tiny-random-flint HF (transformers)", "hf/tiny-random-flint", "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art", "ids",
            [140756, 125813, 22985, 123010, 143067, 136720, 86988, 72881, 88183, 83127, 61195, 133527, 129060, 110388, 65465, 7319, 86512, 79741, 81427, 118452, 20753, 25666, 5229, 81385], "", NeedsCpuFallback: true),
        // Hybrid layer-stack adapter (canonical "hybrid"): Granite-4.0-H (Mamba-2 + NoPE attention + dense / shared-expert MoE), Falcon-H1 (parallel attention + Mamba-2, folded multipliers), Mamba-2.
        new("tiny-random-granite-4.0-h HF (transformers)", "hf/tiny-random-granite-4.0-h", P2, "ids",
            [33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759], ""),
        new("tiny-random-granite-moe-hybrid HF (transformers)", "hf/tiny-random-granite-moe-hybrid", P2, "ids",
            [33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759, 33484, 74759], ""),
        new("tiny-random-falcon-h1 HF (transformers)", "hf/tiny-random-falcon-h1", P2, "ids",
            [31844, 260248, 31844, 260248, 31844, 260248, 31844, 260248, 31844, 260248, 31844, 260248, 31844, 260248, 31844, 260248], ""),
        new("granite-4.0-h-350m HF (transformers)", "hf/granite-4.0-h-350m", P2, "ids",
            [11, 11401, 11, 323, 36105, 13, 11450, 11, 433, 374, 3967, 369, 1202, 18112, 11, 11401], ""),
        new("Falcon-H1-0.5B-Instruct HF (transformers)", "hf/Falcon-H1-0.5B-Instruct", P2, "ids",
            [535, 16379, 535, 823, 6535, 537, 1692, 1179, 774, 26731, 813, 14793, 5897, 823, 774, 2220], ""),
        // LFM2: gated short convolutions + QK-normed attention.
        new("tiny-random-lfm2 HF (transformers)", "hf/tiny-random-lfm2", P2, "ids",
            [11762, 12390, 31881, 32488, 20232, 42609, 52418, 27614, 2051, 53259, 32162, 53259, 32162, 53259, 32162, 53259], ""),
        new("LFM2.5-1.2B-Instruct HF (transformers)", "hf/LFM2.5-1.2B-Instruct", P2, "ids",
            [521, 12688, 521, 810, 11458, 523, 941, 3771, 856, 1236, 779, 6584, 3771, 797, 779, 3374], ""),
        new("LFM2.5-1.2B-Instruct Q4_K_M GGUF (llama.cpp)", "gguf/LFM2.5-1.2B-Instruct/LFM2.5-1.2B-Instruct-Q4_K_M.gguf", "The capital of France is", "text", [],
            " the city of Paris. It is the most populous city in the country and", Tokens: 16),
        // LFM2-MoE: sigmoid-routed experts with a selection bias (shared MoeBlock) after the leading dense layers. The GGUF reference is
        // llama.cpp's Vulkan backend; its CUDA backend picks " Paris" over " the" for the first token (logit margin 0.1). The 13th token
        // is a tie between " the" and " France" (margin 0.05 on both CPU and Vulkan), so the comparison stops before it.
        new("tiny-random-lfm2-moe HF (transformers)", "hf/tiny-random-lfm2-moe", P2, "ids",
            [60355, 59557, 60355, 59557, 60355, 59557, 60355, 59557, 60355, 59557, 60355, 27023, 60355, 27023, 60355, 27023], ""),
        new("LFM2.5-8B-A1B Q4_K_M GGUF (llama.cpp Vulkan)", "gguf/LFM2.5-8B-A1B/LFM2.5-8B-A1B-Q4_K_M.gguf", "The capital of France is", "text", [],
            " the city of Paris. city of Paris is a city in", Tokens: 12),
        // Mamba-1 (per-channel decay, low-rank dt) and Jamba (Mamba-1 + attention + dt/B/C norms).
        new("mamba-130m-hf HF (transformers)", "hf/mamba-130m-hf", "The capital of France is", "ids",
            [275, 253, 2846, 273, 7785, 15, 187, 187, 510, 2846, 273, 7785, 310, 275, 253, 2846], ""),
        new("AI21-Jamba-Reasoning-3B Q4_K_M GGUF (llama.cpp)", "gguf/AI21-Jamba-Reasoning-3B/jamba-reasoning-3b-Q4_K_M.gguf", "The capital of France is", "text", [],
            " Paris.\n\nThe capital of Germany is", Tokens: 9),
        // RWKV-7 G1 (Wkv7 recurrence, value residual, LoRA decay / learning rate / gate); RWKV-6 is covered by LogprobReferenceTests.
        new("rwkv7-g1h-1.5b Q4_K_M GGUF (llama.cpp)", "gguf/rwkv7-g1h-1.5b/rwkv7-g1h-1.5b-20260710-ctx10240.i1-Q4_K_M.gguf", "The capital of France is", "text", [],
            " Paris.\nThe capital of France is Paris.\nThe capital of France is", Tokens: 16),
        new("mamba2-130m-hf HF (transformers)", "hf/mamba2-130m-hf", "The capital of France is", "ids",
            [247, 2846, 273, 253, 5112, 952, 13, 285, 253, 5347, 273, 253, 5112, 4687, 15, 733], ""),
        new("granite-4.0-h-350m Q4_K_M GGUF (llama.cpp)", "gguf/granite-4.0-h-350m/granite-4.0-h-350m-Q4_K_M.gguf", "The capital of France is", "text", [], " Paris.", Tokens: 16),
        new("granite-4.0-h-tiny Q4_K_M GGUF (llama.cpp CPU/Vulkan)", "gguf/granite-4.0-h-tiny/granite-4.0-h-tiny-Q4_K_M.gguf", "The capital of France is", "text", [],
            " Paris. Paris is known for its historical landmarks, such as the Eiffel", Tokens: 16),
        new("Falcon-H1-0.5B-Instruct Q4_K_M GGUF (llama.cpp)", "gguf/Falcon-H1-0.5B-Instruct/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf", "The capital of France is", "text", [],
            " Paris. Paris is known for its iconic landmarks such as the Eiffel", Tokens: 16),
        // Qwen3.5/3.6/3.8 MoE hybrids: gated delta + gated attention layers, softmax top-k experts plus a sigmoid-gated shared expert
        // (per-expert tensors stacked, or fused 3-D gate_up tensors for 3.8). References: tools/hf_greedy_reference.py; the 3.6
        // checkpoint nests its text model three times, which the tool collapses (transformers alone would initialize it randomly).
        new("tiny-random-qwen3.5-moe HF (transformers)", "hf/tiny-random-qwen3.5-moe", "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art", "ids",
            [132029, 22595, 212119, 133773, 4415, 198543, 219592, 20547, 220921, 172932, 151129, 212119, 133773, 239941, 239941], ""),
        new("tiny-random-qwen3.6-moe HF (transformers)", "hf/tiny-random-qwen3.6-moe", "The capital of France is", "ids",
            [171219, 28127, 89241, 147994, 147994, 194404, 115056, 51300, 108175, 25743, 24907, 133054, 2637, 51409, 225298, 94434], ""),
        new("tiny-random-qwen3.8-moe HF (transformers)", "hf/tiny-random-qwen3.8-moe", "The capital of France is", "ids",
            [102419, 116372, 131665, 29461, 41257, 39053, 42030, 35046, 3484, 115260, 59741, 160416, 28413, 208877, 186922, 189616], ""),
        // Qwen3-Next: fused in_proj_qkvz / in_proj_ba, value heads grouped by key head, plain partial RoPE.
        new("tiny-random-qwen3-next-moe HF (transformers)", "hf/tiny-random-qwen3-next-moe", "The capital of France is", "ids",
            [32652, 49041, 83843, 51624, 109256, 116828, 29463, 49041, 22433, 245, 143247, 28709, 49988, 109256, 51255, 117540], ""),
        // Qwen3.8-Flash-Next (qwen4_exp): hyper-connections, hashed n-gram embedding (the second prompt resets the n-gram context at
        // <|endoftext|>), QSA indexer. The tiny n-gram table has 1-wide rows, which the Vulkan gather leaves to the CPU.
        new("tiny-random-qwen3.8-flash-next HF (transformers)", "hf/tiny-random-qwen3.8-flash-next", "The capital of France is", "ids",
            [144327, 144327, 225506, 211030, 97146, 233136, 9370, 73769, 109242, 214491, 226975, 16650, 238421, 24763, 218821, 87098], "", NeedsCpuFallback: true),
        new("tiny-random-qwen3.8-flash-next HF, end-of-text in the prompt (transformers)", "hf/tiny-random-qwen3.8-flash-next", "Hi there<|endoftext|>The capital of France is", "ids",
            [144327, 144327, 144327, 225506, 128955, 40177, 244551, 164360, 41110, 144327, 41110, 144327, 144327, 144327, 225506, 128955], "", NeedsCpuFallback: true),
        // Qwen3-VL text model (interleaved M-RoPE and DeepStack reduce to the dense decoder for text-only prompts).
        new("Qwen3-VL-2B-Instruct HF (transformers)", "hf/Qwen3-VL-2B-Instruct", "The capital of France is", "ids",
            [12095, 11, 323, 279, 6722, 315, 17689, 374, 24081, 13, 1416, 279, 6722, 315, 9625, 374], ""),
        new("Qwen3-VL-2B-Instruct Q4_K_M GGUF (llama.cpp)", "gguf/Qwen3-VL-2B-Instruct/Qwen3VL-2B-Instruct-Q4_K_M.gguf", "The capital of France is", "text", [],
            " Paris, and the capital of Germany is Berlin. If the capital of France is", Tokens: 16),
        // Qwen3.6-35B-A3B (qwen35moe, 256 experts top-8) and Qwen3-Coder-Next (qwen3next, 512 experts top-10, MXFP4 shared expert).
        new("Qwen3.6-35B-A3B UD-Q4_K_M GGUF (llama.cpp)", "gguf/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "The capital of France is", "text", [],
            " Paris, a city renowned for its iconic landmarks such as the Eiffel Tower,\n\n", Large: true, Tokens: 16),
        new("Qwen3-Coder-Next Q4_K_M GGUF (llama.cpp)", "gguf/Qwen3-Coder-Next/Qwen3-Coder-Next-Q4_K_M.gguf", "The capital of France is", "text", [],
            " Paris. The capital of Germany is Berlin. The capital of Italy is Rome.\n\n", Large: true, Tokens: 16),
        // Qwen3.8-Flash-Next (qwen4exp, 111 GB): the layers that do not fit on the GPU run on the CPU; the 28.8 GB n-gram table
        // exercises weight rows beyond 4 GiB.
        new("Qwen3.8-Flash-Next UD-Q4_K_XL GGUF (llama.cpp)",
            "gguf/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", "The capital of France is", "text", [],
            " Paris. The capital of Germany is Berlin. The capital of Italy is Rome.\n\n", NeedsCpuFallback: true, Large: true, Tokens: 16),
        // Generic-decoder families (waves 0 and 2 of docs/model-support.md section 5).
        .. DecoderFamilyCases.All,
        // Routed-MoE families, gpt-oss and BitNet (waves 3 and 6).
        .. RoutedMoeFamilyCases.All,
    ];

    private static (LlmModel Model, DeviceSlot[] Slots) Load(string path, bool vulkan, bool cpuFallback)
    {
        DeviceSlot slot, fallback = null!;
        if (vulkan)
        {
            var p = new VulkanProvider();
            if (!p.CheckAvailability().IsAvailable)
            {
                throw new SkipException("Vulkan is not available");
            }

            var devs = p.GetDevices();
            var dev = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_VK_DEVICE") is { } e && int.TryParse(e, out int i) && i < devs.Count ? devs[i]
                : devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault() ?? throw new SkipException("no Vulkan device");
            slot = new DeviceSlot { Device = p.OpenDevice(dev) };
        }
        else
        {
            var p = new CpuExecutionProvider();
            slot = new DeviceSlot { Device = p.OpenDevice(p.GetDevices()[0]) };
        }
        if (vulkan && cpuFallback)
        {
            var c = new CpuExecutionProvider();
            fallback = new DeviceSlot { Device = c.OpenDevice(c.GetDevices()[0]) };
        }
        var model = LlmModel.Load(path, new LlmLoadOptions { Devices = [slot], Fallback = vulkan && cpuFallback ? fallback : null, ContextLength = 256, MaxBatchTokens = 256 });
        return (model, vulkan && cpuFallback ? [slot, fallback] : [slot]);
    }

    private sealed class SkipException(string reason) : Exception(reason);

    private static (int[] Ids, string Text) Generate(string path, string prompt, int n, bool vulkan, bool cpuFallback)
    {
        var (model, slots) = Load(path, vulkan, cpuFallback);
        try
        {
            using var session = model.CreateSession(WellKnown.DecodeGraph);
            var gen = new Generator(session, model.Tokenizer);
            var toks = gen.Generate(prompt, new GenerationOptions { MaxNewTokens = n, Sampling = SamplingOptions.Greedy }).ToList();
            return ([.. toks.Select(t => t.Id)], string.Concat(toks.Select(t => t.Text)));
        }
        finally
        {
            model.Dispose();
            foreach (var s in slots)
            {
                s.Device.Dispose();
            }
        }
    }

    private void Run(string name, bool vulkan)
    {
        var c = Cases.Single(x => x.Name == name);
        string path = System.IO.Path.Combine(Home, "models", c.Path);
        if (!File.Exists(path) && !Directory.Exists(path))
        {
            Assert.Skip($"{c.Name}: {path} not present");
            return;
        }
        if (vulkan && c.Large && Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_LARGE") is null)
        {
            Assert.Skip("large model: set SHARPLIGHT_TEST_LARGE=1 to run it");
            return;
        }
        if (!vulkan && c.Large)
        {
            Assert.Skip("large model is only run on the GPU");
            return;
        }
        (int[] ids, string text) result;
        try { result = Generate(path, c.Prompt, c.Kind == "ids" ? c.Ids.Length : c.Tokens, vulkan, c.NeedsCpuFallback); }
        catch (SkipException s)
        {
            Assert.Skip(s.Message);
            return;
        }
        if (c.Kind == "ids")
        {
            Assert.Equal(c.Ids, result.ids);
        }
        else
        {
            Assert.Equal(c.Text.TrimEnd(), result.text.TrimEnd());
        }
    }

    [Theory, MemberData(nameof(CaseNames))]
    public void Generation_matches_reference_on_cpu(string name) => Run(name, vulkan: false);

    [Theory, MemberData(nameof(CaseNames))]
    public void Generation_matches_reference_on_vulkan(string name) => Run(name, vulkan: true);
}
