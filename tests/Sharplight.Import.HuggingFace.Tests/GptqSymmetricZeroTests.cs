using System.Text.Json;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// The zero point of symmetric GPTQ checkpoints: 2^(bits-1) whatever <c>qzeros</c> stores, as vLLM's GPTQ-Marlin kernels read it.
/// Qwen's official GPTQ-Int4 checkpoints (Qwen3.5-122B-A10B) store 8 in the original format, which the plus-one rule would read as 9;
/// an asymmetric checkpoint keeps the stored value plus one.
/// </summary>
public sealed class GptqSymmetricZeroTests : IDisposable
{
    private const int Rows = 8;
    private const int Columns = 128;

    private readonly string _directory = Path.Combine(Path.GetTempPath(), "sharplight-gptq-sym-" + Guid.NewGuid().ToString("N"));

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Theory]
    [InlineData(true, 8f)]
    [InlineData(false, 9f)]
    public void ZeroPointFollowsTheSymmetricFlag(bool symmetric, float expectedZero)
    {
        Directory.CreateDirectory(_directory);
        var rng = new Random(3);
        byte[] qweight = new byte[Columns / 8 * Rows * 4];
        rng.NextBytes(qweight);
        byte[] qzeros = [.. Enumerable.Repeat((byte)0x88, Rows / 8 * 4)];
        byte[] scales = [.. Enumerable.Range(0, Rows).SelectMany(_ => BitConverter.GetBytes((Half)0.5f))];
        string path = Path.Combine(_directory, "model.safetensors");
        new SafetensorsWriter()
            .Add("layer.qweight", "I32", [Columns / 8, Rows], qweight)
            .Add("layer.qzeros", "I32", [1, Rows / 8], qzeros)
            .Add("layer.scales", "F16", [1, Rows], scales)
            .Write(path);
        string flag = symmetric ? "true" : "false";
        using var settings = JsonDocument.Parse($$"""{"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": {{flag}}}""");

        using var file = SafetensorsFile.Open(path);
        var layout = Assert.Single(HfGptqLayouts.Find(settings.RootElement, file.Tensors, []));
        byte[] codes = new byte[Columns];
        float[] rowScales = new float[1];
        float[] rowZeros = new float[1];
        for (long row = 0; row < Rows; row++)
        {
            layout.DecodeRow(row, codes, rowScales, rowZeros);
            Assert.Equal(expectedZero, rowZeros[0]);
            Assert.Equal(0.5f, rowScales[0]);
        }
    }
}
