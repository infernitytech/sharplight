using System.Text.Json;
using Xunit;

namespace Sharplight.Import.PyTorch.Tests;

/// <summary>
/// Checkpoints written by torch.save itself (tools/gen-torch-fixtures.py, torch 2.15): the zip and the legacy stream format, a module
/// state dict and a training checkpoint. Every tensor's bytes must equal the bytes torch holds.
/// </summary>
public sealed unsafe class TorchSavedFixtureTests
{
    private static readonly string s_data = Path.Combine(AppContext.BaseDirectory, "Data");

    public static TheoryData<string> Files() => ["zip.pt", "legacy.pt", "module.bin", "training.ckpt"];

    private static TorchDType DType(string name) => name switch
    {
        "float32" => TorchDType.Float32,
        "float16" => TorchDType.Float16,
        "bfloat16" => TorchDType.BFloat16,
        "float64" => TorchDType.Float64,
        "int64" => TorchDType.Int64,
        "int32" => TorchDType.Int32,
        "int16" => TorchDType.Int16,
        "int8" => TorchDType.Int8,
        "uint8" => TorchDType.UInt8,
        "bool" => TorchDType.Bool,
        _ => throw new InvalidOperationException(name),
    };

    [Theory]
    [MemberData(nameof(Files))]
    public void TensorsAreBitIdenticalToTorch(string fileName)
    {
        using var expectedDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(s_data, "expected.json")));
        var expected = expectedDoc.RootElement.GetProperty(fileName);
        using var file = TorchCheckpointFile.Open(Path.Combine(s_data, fileName));

        Assert.Equal(expected.EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal), file.Tensors.Keys.Order(StringComparer.Ordinal));
        foreach (var property in expected.EnumerateObject())
        {
            var tensor = file.Tensors[property.Name];
            Assert.Equal(DType(property.Value.GetProperty("dtype").GetString()!), tensor.DType);
            Assert.Equal(property.Value.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()), tensor.Shape);
            byte[] want = Convert.FromHexString(property.Value.GetProperty("hex").GetString()!);
            Assert.Equal(want.Length, tensor.ByteLength);
            Assert.Equal(want, new ReadOnlySpan<byte>(file.BasePointer + tensor.FileOffset, want.Length).ToArray());
        }
    }

    [Fact]
    public void TrainingCheckpointUnwrapsStateDict()
    {
        using var file = TorchCheckpointFile.Open(Path.Combine(s_data, "training.ckpt"));
        Assert.Contains("0.weight", file.Tensors.Keys);
        Assert.DoesNotContain("epoch", file.Tensors.Keys);
    }

    [Fact]
    public void DisposedFileCannotBeRead()
    {
        var file = TorchCheckpointFile.Open(Path.Combine(s_data, "zip.pt"));
        file.Dispose();
        Assert.Throws<ObjectDisposedException>(() => (nint)file.BasePointer);
    }
}
