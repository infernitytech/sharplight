using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// The quantized formats of the second quantization round (tests/Sharplight.Families.Tests/tools/mk_tiny_quantized_formats.py,
/// references from hf_quant_reference_formats.py), checked like <see cref="QuantizedImportTests"/>: every converted weight against
/// the library's own dequantization, bit for bit, on the CPU and on Vulkan, with the expected quantized type. Skips when the
/// checkpoints are absent.
/// </summary>
public sealed class QuantizedFormatsImportTests(ITestOutputHelper output)
{
    /// <summary>Checkpoint directory (under ~/models/hf/quant) and the type its linear weights must convert to.</summary>
    public static TheoryData<string, string> Checkpoints() => new()
    {
        { "tiny/tiny-llama-mlx-2bit-g64", "q4_1" },
        { "tiny/tiny-llama-mlx-3bit-g64", "q4_1" },
        { "tiny/tiny-llama-mlx-5bit-g64", "q5_1" },
        { "tiny/tiny-llama-mlx-6bit-g64", "bf16" },
        { "tiny/tiny-llama-mlx-mxfp4", "mxfp4" },
        { "tiny/tiny-llama-mlx-nvfp4", "nvfp4" },
        { "tiny/tiny-llama-mlx-mxfp8", "f8e4m3_b32" },
        { "tiny/tiny-llama-autoround-gptq-4bit-sym", "q4z_g128" },
        { "tiny/tiny-llama-autoround-gptq-4bit-asym-g32", "q4z_g32" },
        { "tiny/tiny-llama-autoround-gptq-2bit-g64", "q4z_g32" },
        { "tiny/tiny-llama-autoround-awq-4bit", "q4z_g128" },
        { "tiny/tiny-llama-eetq-int8", "q8z_g128" },
        { "tiny/tiny-llama-hqq-4bit-g64", "q4z_g32" },
        { "tiny/tiny-llama-hqq-8bit-g128", "q8z_g128" },
        { "tiny/tiny-llama-hqq-3bit-g64", "q4z_g32" },
        { "tiny/tiny-llama-hqq-2bit-g32", "q4z_g32" },
        { "tiny/tiny-llama-hqq-1bit-g64", "q4z_g32" },
        { "tiny/tiny-llama-hqq-4bit-g64-axis0", "bf16" },
    };

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void DequantizesLikeTheLibraryOnCpu(string checkpoint, string type) => QuantizedImportTests.Check(output, checkpoint, type, vulkan: false);

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void DequantizesLikeTheLibraryOnVulkan(string checkpoint, string type) => QuantizedImportTests.Check(output, checkpoint, type, vulkan: true);
}
