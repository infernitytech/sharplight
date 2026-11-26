using Sharplight.Cpu;
using Sharplight.Cuda;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Logits-level checks for models whose greedy text is too fragile to pin (near-tied top tokens): the log-probabilities of the reference's top tokens,
/// captured from llama.cpp b11177 (CPU, weights re-quantized to F16 from the checked-in Q8_0 file so no activation quantization noise), must be matched within
/// the quantization noise of the Q8_0 file the tests load. Truncated real checkpoints compare with transformers in float32 on the file's own
/// dequantized weights instead (no quantization noise in the reference, so a tighter tolerance).
/// </summary>
public class LogprobReferenceTests
{
    private const double DefaultTolerance = 0.25;
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    /// <summary>A prompt and the reference log-probability of some next-token candidates.</summary>
    /// <param name="Name">The case name.</param>
    /// <param name="Path">The model below ~/models.</param>
    /// <param name="Prompt">The prompt, encoded with the model's special tokens.</param>
    /// <param name="Expected">Reference log-probabilities of next-token candidates.</param>
    /// <param name="Tolerance">Largest allowed difference.</param>
    /// <param name="CudaDisabled">Device features turned off on CUDA (paths that quantize activations, which the reference does not).</param>
    public sealed record Case(string Name, string Path, string Prompt, (int Id, double LogProb)[] Expected, double Tolerance = DefaultTolerance,
        string[]? CudaDisabled = null);

    // The Super-VL release declares W8A8 FP8 (static per-tensor input scales) for its Mamba, attention and LM-head projections, so CUDA's FP8
    // GEMMs quantize their activations to E4M3 as the format allows. This 8-layer cut is that sensitive to FP8 activation rounding (likely a near
    // tie in its 22-of-512 expert selection) that it moves 0.17-0.19 with either rounding: ours (dynamic per-128-value scales) and the release's own (static scales, simulated
    // in the reference with tools/hf-logprob-reference.py --fp8-input-scales: (99093, -4.491), (35192, -4.929), (26642, -5.179),
    // (73731, -5.241), (26114, -5.522)) part by up to 0.31, so the case compares the products without FP8 activations. Its W4A4 NVFP4
    // experts keep the native FP4 GEMM (0.06 off). Bolt's NVFP4 projections are weight-only (W4A16) and never take it.
    private static readonly string[] Fp8Activations =
        [DeviceFeatures.BlockScaledFp8, DeviceFeatures.NativeGemmFp8, DeviceFeatures.MatrixFp8, DeviceFeatures.Float8];

    // Inkling-2-Mini's parameter groups declare NVFP4 activations for its NVFP4 experts and MXFP8 ones for its MXFP8 experts; transformers'
    // reference dequantizes the weights only, so its case compares the products without FP4 or FP8 activations.
    private static readonly string[] Fp4AndFp8Activations = [DeviceFeatures.BlockScaledFp4, .. Fp8Activations];

    private static readonly Case[] Cases =
    [
        // RWKV-6 World 1.6B: llama.cpp F16 (the Q8_0 model flips the near-tied " the" / " France" continuation).
        new("rwkv6 'The'", "gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf", "The", [(267, -3.768), (38499, -4.545), (31461, -5.077), (265, -5.107)]),
        new("rwkv6 'The capital of France is'", "gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf", "The capital of France is", [(37138, -0.773), (22590, -2.585), (4596, -2.869), (22187, -3.169)]),
        new("rwkv6 'The capital of France is Paris.\\nThe capital of'", "gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf", "The capital of France is Paris.\nThe capital of",
            [(44312, -2.049), (22590, -2.134), (50275, -2.527), (36847, -2.748)]),
        // DeepSeek-V3.1 (unsloth Q4_K_M) cut to its first 6 layers (3 dense, 3 MoE with 256 real experts; tools/truncate-gguf.py), the
        // prompt at which the full model's greedy text and llama.cpp's part (a near tie of " Spain" / " France"). Reference: transformers'
        // DeepseekV3 in float32 on the file's dequantized weights (tools/latent-moe-gguf-reference.py ... 8 4: the model's group-limited
        // routing, 8 groups with the best 4 kept, which the importer defaults to for DeepSeek-V3 files without expert_group_count). Ours is
        // within 0.03 on CPU; without the groups (as llama.cpp b11177 routes this file) token 50365 is 0.19 off and the argmax differs.
        new("deepseek-v3.1 6 layers 'The capital of France is Paris. The capital of'", "gguf/DeepSeek-V3.1-trunc/DeepSeek-V3.1-Q4_K_M-L6.gguf",
            "The capital of France is Paris. The capital of", [(81661, -3.608), (50365, -3.633), (104323, -4.217), (119996, -4.387), (74218, -4.778)],
            Tolerance: 0.1),
        // GLM-5.2 (unsloth UD-Q2_K_XL) cut to its first 6 layers (3 dense, 3 MoE, DSA indexer in its dense regime), at the prompt where the
        // full model's greedy text and llama.cpp's part. Reference: transformers' GlmMoeDsa in float32 on the file's dequantized weights
        // (tools/latent-moe-gguf-reference.py). Ours is within 0.04 on CPU; llama.cpp b11177 within 0.06.
        new("glm-5.2 6 layers 'The capital of France is Paris. Distance ... while direct flight time'", "gguf/GLM-5.2-trunc/GLM-5.2-UD-Q2_K_XL-L6.gguf",
            "The capital of France is Paris. Distance from Paris to Marseille is 661 km, while direct flight time",
            [(21158, -6.023), (79386, -6.209), (52628, -6.407), (7588, -6.598), (4740, -6.609)], Tolerance: 0.1),
        // Kimi-K3 (unsloth UD-Q1_0) cut to its first 4 layers (3 Kimi delta-attention layers, one latent-attention layer, real latent MoE),
        // its narrow IQ1 experts widened to IQ1_S (tools/truncate-gguf-remote.py --widen-narrow-iq1) so that llama.cpp b11177 reads it.
        // Reference: llama.cpp b11177 with the prompt decoded token by token (CPU; its batched prefill is within 0.04 of that).
        new("kimi-k3 4 layers 'The capital of France is Paris. The capital of Germany is'", "gguf/Kimi-K3-trunc/Kimi-K3-UD-Q1_0-L4-iq1s.gguf",
            "The capital of France is Paris. The capital of Germany is",
            [(36356, -5.422), (140580, -5.526), (24917, -5.601), (127274, -5.689), (19320, -5.871)], Tolerance: 0.12),
        // MiniMax-M3.1-Flash (the release: MXFP8 dense weights, E8M0 scale per 32 values; NVFP4 compressed-tensors experts) cut to its first 4
        // layers (3 dense, 1 MoE) and 32 of 128 experts with the router rows to match (tools/truncate-hf-remote.py ... --layers 4 --experts 32,
        // pinned revision ce7f4df). Reference: transformers' MiniMaxM3SparseForConditionalGeneration in float32 on the dequantized weights
        // (tools/dequantize-hf-checkpoint.py, then tools/hf-logprob-reference.py).
        new("minimax-m3.1-flash 4 layers 32 experts 'The capital of France is Paris. The capital of Germany is'", "hf/MiniMax-M3.1-Flash-L4-E32",
            "The capital of France is Paris. The capital of Germany is",
            [(3051, -4.019), (44525, -4.447), (97269, -5.273), (74406, -5.410), (14610, -5.450)], Tolerance: 0.1),
        // MiMo-V2.6-Pro-RL (the release: block FP8 dense weights, the fused qkv_proj stored as 8 slices [Q_i | K_i | V_i] quantized one by one,
        // MXFP4 experts) cut to its first 3 layers (one dense global layer, two sliding MoE layers) and 32 of 384 experts (pinned revision
        // a066ba0). Reference: the release's remote code (modeling_mimo_v2.py; two masking arguments renamed for transformers 5.24) in float32
        // on the dequantized weights, the qkv slices reordered to [Q | K | V] as vLLM's mimo_v2 loader reads them
        // (tools/dequantize-hf-checkpoint.py --qkv-slices 8:3072:192:128).
        new("mimo-v2.6-pro-rl 3 layers 32 experts 'The capital of France is Paris. The capital of Germany is'", "hf/MiMo-V2.6-Pro-RL-L3-E32",
            "The capital of France is Paris. The capital of Germany is",
            [(132126, -0.980), (118364, -1.013), (143401, -3.387), (117170, -3.421), (138523, -3.519)], Tolerance: 0.1),
        // DeepSeek-V4.1-Flash (the release: FP8 E4M3 weights with E8M0 scales per 32 x 32 block, MXFP4 experts, E4M3 Engram tables with an
        // E8M0 scale per 32 values) cut to its first 3 layers (two sliding-window layers, one compressing pairs of positions with its own
        // indexer; the Engram memory of layer 1) and 32 of 384 experts, the Engram buckets of layer 1 rebuilt above 100000 instead of
        // 16000000 with the table cut to their 2403994 rows (tools/truncate-hf-remote.py ... --set text_config.engram_vocab_size=100000,
        // pinned revision 314113f). Reference: the release's own inference/model.py on its convert.py output, its tilelang kernels replaced
        // by torch code with the same arithmetic, in float32 without the simulated activation quantization (tools/deepseek-v41-reference.py).
        new("deepseek-v4.1-flash 3 layers 32 experts 'The capital of France is Paris. The capital of Germany is'", "hf/DeepSeek-V4.1-Flash-L3-E32",
            "The capital of France is Paris. The capital of Germany is",
            [(98883, -2.142), (116971, -2.212), (32345, -3.825), (88049, -3.885), (101275, -3.975)], Tolerance: 0.1),
        // NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB (the release: ModelOpt MIXED_PRECISION with ggml blocks packed in safetensors, Q8_0 attention
        // and latent projections, IQ2_XXS experts, NVFP4 Mamba projections) cut to its first 8 layers (4 Mamba-2, 3 latent MoE with 240
        // experts of the Puzzle configs, 1 attention; tools/truncate-hf-remote.py ... --layers 8, pinned revision e0590ed). Reference: the
        // release's remote code (modeling_nemotron_h_puzzle.py, torch path; mamba_ssm's gated RMS norm from its torch reference) in float32
        // on the dequantized weights (gguf-py's dequantizers for the ggml blocks; tools/dequantize-hf-checkpoint.py --rename
        // language_model.backbone.=language_model.model.).
        new("nemotron-3.5-bolt 8 layers 'The capital of France is Paris. The capital of Germany is'", "hf/Nemotron-3.5-Bolt-74B-A9B-24GB-L8",
            "The capital of France is Paris. The capital of Germany is",
            [(35192, -4.270), (99093, -4.419), (17893, -4.916), (75560, -5.583), (99868, -5.612)], Tolerance: 0.1),
        // NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (the release: ModelOpt NVFP4 experts with a second-level scale per expert, FP8
        // per-tensor attention, Mamba and LM-head weights) cut to its first 8 layers (4 Mamba-2, 3 latent MoE with all 512 experts,
        // 1 attention; pinned revision ae13683). Reference: the release's remote code (modeling_nemotron_h_omni.py, torch path) in
        // float32 on the dequantized weights.
        new("nemotron-3.5-super-vl-nvfp4 8 layers 'The capital of France is Paris. The capital of Germany is'", "hf/Nemotron-3.5-Super-VL-120B-A12B-NVFP4-L8",
            "The capital of France is Paris. The capital of Germany is",
            [(99093, -4.391), (35192, -4.766), (26642, -5.298), (73731, -5.329), (26114, -5.329)], Tolerance: 0.1,
            CudaDisabled: Fp8Activations),
        // Inkling-2-Mini (the release: ModelOpt parameter groups, NVFP4 experts in layers 0-19 with a per-expert amax / outer-denominator factor,
        // MXFP8 experts in layers 20-23, BF16 elsewhere) with all 24 layers (the loop over layers 6-17 run 3 times) and 32 of 128 experts with
        // the router rows to match (tools/truncate-hf-remote.py ... --layers 24 --experts 32 --rows 'mlp\.experts\.w(13|2)_weight(_scale|_amax)?$=32',
        // pinned revision c9d0283). Reference: transformers 5.24.1's Inkling2LoopForConditionalGeneration in float32, the experts dequantized
        // by its own conversion op (Inkling2LoopDequantizeExperts). CUDA without its native FP4 / FP8 paths: they quantize the activations (the
        // release's hf_quant_config asks for NVFP4 / MXFP8 activations, which transformers' dequantized reference does not apply; 1.1 off).
        new("inkling-2-mini 24 layers 32 experts 'The capital of France is Paris. The capital of Germany is'", "hf/Inkling-2-Mini-E32",
            "The capital of France is Paris. The capital of Germany is",
            [(154328, -0.744), (220, -2.491), (149884, -2.943), (175232, -3.013), (145038, -3.412)], Tolerance: 0.1, CudaDisabled: Fp4AndFp8Activations),
    ];

    public static IEnumerable<TheoryDataRow<string>> CaseNames() => Cases.Select(c => new TheoryDataRow<string>(c.Name));

    private enum Device
    {
        Cpu,
        Vulkan,
        Cuda,
    }

    private static void Run(string name, Device kind)
    {
        var c = Cases.Single(x => x.Name == name);
        string path = System.IO.Path.Combine(Home, "models", c.Path);
        Assert.SkipUnless(File.Exists(path) || Directory.Exists(path), $"{path} not present");
        IExecutionProvider provider = kind switch
        {
            Device.Vulkan => new VulkanProvider(),
            Device.Cuda => new CudaProvider(),
            _ => new CpuExecutionProvider(),
        };
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "provider not available");
        var devices = provider.GetDevices();
        var info = kind == Device.Vulkan ? devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0] : devices[0];
        var slot = new DeviceSlot { Device = provider.OpenDevice(info) };
        // The test owns the device (models do not): disposed after the model, so no device outlives the test.
        using var device = slot.Device;
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions
        {
            Devices = [slot], ContextLength = 256, MaxBatchTokens = 64,
            DisabledFeatures = kind == Device.Cuda && c.CudaDisabled is { } disabled ? new HashSet<string>(disabled) : new HashSet<string>(),
        });
        using var session = model.CreateSession(WellKnown.DecodeGraph);
        var logits = session.Process(model.Tokenizer.Encode(c.Prompt, addSpecial: true, parseSpecial: false));
        double max = logits.Max();
        double logSum = Math.Log(logits.Sum(l => Math.Exp(l - max)));
        foreach (var (id, expected) in c.Expected)
        {
            double actual = logits[id] - max - logSum;
            TestContext.Current.TestOutputHelper?.WriteLine($"{c.Name}: token {id} log-probability {actual:F3}, reference {expected:F3}");
            Assert.True(Math.Abs(actual - expected) < c.Tolerance, $"{c.Name}: token {id} log-probability {actual:F3}, reference {expected:F3}");
        }
    }

    [Theory, MemberData(nameof(CaseNames))]
    public void Logprobs_match_reference_on_cpu(string name) => Run(name, Device.Cpu);

    [Theory, MemberData(nameof(CaseNames))]
    public void Logprobs_match_reference_on_vulkan(string name) => Run(name, Device.Vulkan);

    [Theory, MemberData(nameof(CaseNames))]
    public void Logprobs_match_reference_on_cuda(string name) => Run(name, Device.Cuda);
}
