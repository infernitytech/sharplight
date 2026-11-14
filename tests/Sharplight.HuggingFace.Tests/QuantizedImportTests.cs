using System.Runtime.InteropServices;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan;
using Xunit;

namespace Sharplight.HuggingFace.Tests;

/// <summary>
/// Quantized Hugging Face checkpoints made by the quantization libraries themselves (tests/Sharplight.Families.Tests/tools/
/// mk_tiny_quantized.py: llm-compressor, AutoAWQ, bitsandbytes, GPTQ packed following gptqmodel) and small real Hub checkpoints,
/// each with the library's own dequantization of its weights (hf_quant_reference.py, <c>sharplight-reference/</c>): every converted
/// weight is read back through a Gather graph on the CPU and on Vulkan, put back into the checkpoint's column order (act-order
/// weights are stored sorted by group) and rounded to the dtype the library dequantizes to; it must equal the library's values bit
/// for bit (for the formats documented to decode to BF16 at load, bitsandbytes FP4 and LLM.int8: the library's values rounded to
/// BF16), and it must have kept the expected quantized type. Skips when the
/// checkpoints are absent.
/// </summary>
public sealed class QuantizedImportTests(ITestOutputHelper output)
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    /// <summary>Checkpoint directory (under ~/models/hf/quant) and the type its linear weights must convert to.</summary>
    public static TheoryData<string, string> Checkpoints() => new()
    {
        { "tiny/tiny-llama-ct-nvfp4a16", "nvfp4_gs" },
        { "tiny/tiny-llama-ct-mxfp4a16", "mxfp4" },
        { "tiny/tiny-llama-ct-w4a16-g128", "q4z_g128" },
        { "tiny/tiny-llama-ct-w4a16-actorder", "q4z_g128" },
        { "tiny/tiny-llama-ct-w4a16-asym-g32", "q4z_g32" },
        { "tiny/tiny-llama-ct-w8a16-channel", "q8z_g128" },
        { "tiny/tiny-llama-ct-w8a8-int", "q8z_g128" },
        { "tiny/tiny-llama-awq-gemm", "q4z_g128" },
        { "tiny/tiny-llama-awq-gemv", "q4z_g128" },
        { "tiny/tiny-llama-bnb-nf4", "nf4_b64" },
        { "tiny/tiny-llama-bnb-nf4-double", "nf4_b64" },
        { "tiny/tiny-llama-bnb-fp4", "bf16" },
        { "tiny/tiny-llama-bnb-int8", "bf16" },
        { "tiny/tiny-llama-gptq-4bit-g128-sym", "q4z_g128" },
        { "tiny/tiny-llama-gptq-4bit-g128-actorder", "q4z_g128" },
        { "tiny/tiny-llama-gptq-4bit-g32-v2", "q4z_g32" },
        { "tiny/tiny-llama-gptq-3bit-g128", "q4z_g128" },
        { "tiny/tiny-llama-gptq-8bit-channel", "q8z_g128" },
        { "tiny/tiny-llama-quark-mxfp4", "mxfp4" },
        { "tiny/tiny-llama-mlx-4bit-g64", "q4_1" },
        { "tiny/tiny-llama-mlx-4bit-g32", "q4_1" },
        { "tiny/tiny-llama-mlx-8bit-g64", "bf16" },
        { "NVFP4_Qwen3-0.6B-FP4", "nvfp4_gs" },
        { "mgoin_Qwen3-0.6B-NVFP4", "nvfp4_gs" },
    };

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void DequantizesLikeTheLibraryOnCpu(string checkpoint, string type) => Check(checkpoint, type, vulkan: false);

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void DequantizesLikeTheLibraryOnVulkan(string checkpoint, string type) => Check(checkpoint, type, vulkan: true);

    private void Check(string checkpoint, string type, bool vulkan)
    {
        string directory = Path.Combine(Home, "models", "hf", "quant", checkpoint);
        string referenceDirectory = Path.Combine(directory, "sharplight-reference");
        Assert.SkipUnless(Directory.Exists(referenceDirectory), $"{referenceDirectory} not present");
        var device = OpenDevice(vulkan);
        Assert.SkipWhen(device is null, "provider not available");

        using var config = HfConfig.Load(directory);
        using var model = SafetensorsCheckpoint.Open(directory);
        using var reference = SafetensorsCheckpoint.Open(referenceDirectory);
        var resolved = HfQuantizedWeights.Resolve(config, model.Tensors);
        int checkedWeights = 0;
        foreach (var (name, expected) in reference.Tensors)
        {
            if (!name.StartsWith("dequant.", StringComparison.Ordinal))
            {
                continue;
            }

            string weightName = name["dequant.".Length..] + ".weight";
            var weight = resolved[weightName].ToWeightInfo();
            Assert.Equal(DataType.Get(type), weight.Type);
            float[] actual = Decode(weight, device!);
            float[] library = HfTransforms.DecodeF32(expected.ToWeightInfo());
            Assert.Equal(library.Length, actual.Length);
            long columns = weight.Shape[-1].Fixed;
            int mismatches = 0;
            string first = "";
            for (long i = 0; i < actual.LongLength; i++)
            {
                long row = i / columns;
                long column = weight.InputOrder is { } order ? row * columns + order[i % columns] : i;
                float value = Round(actual[i], expected.DType);
                // Formats decoded to BF16 at load (bitsandbytes FP4 and LLM.int8) are the library's values rounded to BF16.
                float want = weight.Type == DataType.BF16 ? Round(library[column], "BF16") : library[column];
                if (BitConverter.SingleToInt32Bits(value) != BitConverter.SingleToInt32Bits(want) && !(value == 0 && want == 0))
                {
                    if (mismatches++ == 0)
                    {
                        first = $"[{row}, {column % columns}] {value:R} (f32 {actual[i]:R}), library {want:R}";
                    }
                }
            }

            Assert.True(mismatches == 0, $"{checkpoint} {weightName} ({weight.Type}): {mismatches} values differ from the library's {expected.DType} dequantization, first {first}");
            checkedWeights++;
        }

        output.WriteLine($"{checkpoint}: {checkedWeights} weights bit-exact ({type})");
        Assert.True(checkedWeights > 0, "the reference holds no dequantized weights");
    }

    /// <summary><paramref name="value"/> rounded to the library's dtype, as the library's f16/bf16 dequantization rounds it.</summary>
    private static float Round(float value, string dtype) => dtype switch
    {
        "F16" => (float)(Half)value,
        "BF16" => BitConverter.Int32BitsToSingle(HfFp8Codec.ToBFloat16(value) << 16),
        _ => value,
    };

    private static IComputeDevice? OpenDevice(bool vulkan)
    {
        IExecutionProvider provider = vulkan ? new VulkanProvider() : new CpuExecutionProvider();
        if (!provider.CheckAvailability().IsAvailable)
        {
            return null;
        }

        var devices = provider.GetDevices();
        var info = vulkan ? devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0] : devices[0];
        return provider.OpenDevice(info);
    }

    /// <summary>Every row of <paramref name="weight"/> in f32, gathered by the device's own weight decoder.</summary>
    private static float[] Decode(WeightInfo weight, IComputeDevice device)
    {
        long rows = weight.Shape[0].Fixed;
        long columns = weight.Shape[-1].Fixed;
        var builder = new GraphBuilder("dequantize", name => name == weight.Name ? weight : null);
        var indices = builder.Input("rows", InputRole.Other, DataType.I32, new Shape([Dim.Sym("T")]));
        builder.Output("y", builder.Gather(builder.Weight(weight.Name), indices));
        using var compiled = CompiledModel.Compile(builder.Build(), new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions { MaxTokens = rows, ContextLength = 16 },
            ReservePerDevice = 0,
        });
        int[] all = [.. Enumerable.Range(0, (int)rows)];
        compiled.SetInput(indices, MemoryMarshal.AsBytes(all.AsSpan()));
        var parameters = new RunParameters();
        parameters.Symbols["T"] = rows;
        parameters.Scalars["n_past"] = 0;
        compiled.Run(parameters);
        var result = new float[rows * columns];
        compiled.GetOutput<float>("y", result);
        return result;
    }
}
