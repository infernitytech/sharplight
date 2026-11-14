using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Gguf.Tests;

/// <summary>
/// <see cref="GgufSupport"/> answers from the registered mappings, and <see cref="GgufHeaderProbe.ReadTensorTypes"/> reads the tensor
/// table from the start of a file, as a range request before a download fetches it.
/// </summary>
public class GgufSupportTests
{
    private static readonly string s_models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf");

    [Theory]
    [InlineData("llama", SupportLevel.Supported)]
    [InlineData("qwen3", SupportLevel.Supported)]
    [InlineData("QWEN35", SupportLevel.Supported)]
    [InlineData("qwen4exp", SupportLevel.Partial)]
    [InlineData("hrm_text", SupportLevel.Unsupported)]
    [InlineData("", SupportLevel.Unsupported)]
    [InlineData(null, SupportLevel.Unsupported)]
    public void ArchitectureSupport_FollowsTheMappings(string? architecture, SupportLevel expected)
    {
        Assert.Equal(expected, GgufSupport.GetArchitectureSupport(architecture));
        Assert.Equal(expected == SupportLevel.Partial, GgufSupport.GetPartialSupportNote(architecture) is not null);
    }

    [Fact]
    public void Architectures_ListEveryMappedName_AndTheProjector()
    {
        Assert.Contains("llama", GgufSupport.Architectures);
        Assert.Contains("clip", GgufSupport.Architectures);
        Assert.True(GgufSupport.IsProjector("clip"));
        Assert.False(GgufSupport.IsProjector("llama"));
        Assert.False(GgufSupport.IsProjector("hrm_text"));
    }

    [Fact]
    public void TensorTypes_KnownAndUnknownIds()
    {
        Assert.Equal(DataType.Q4_K, GgufSupport.GetTensorType(12));
        Assert.Equal(DataType.Q8_K, GgufSupport.GetTensorType(15));
        Assert.Null(GgufSupport.GetTensorType(31));
        Assert.Null(GgufSupport.GetTensorType(99));
    }

    [Fact]
    public void ReadTensorTypes_CountsTheTableWhenThePrefixCoversIt()
    {
        using var stream = new MemoryStream();
        new GgufWriter()
            .AddMetadata("general.architecture", "llama")
            .AddMetadata("tokenizer.ggml.tokens", Enumerable.Range(0, 2000).Select(i => $"token-{i}").ToArray())
            .AddTensor("a", DataType.F32, new byte[16], 4)
            .AddTensor("b", DataType.Q8_K, new byte[292], 256)
            .AddTensor("c", DataType.F32, new byte[16], 4)
            .Write(stream);
        var bytes = stream.ToArray();

        var types = GgufHeaderProbe.ReadTensorTypes(bytes);
        Assert.NotNull(types);
        Assert.Equal(2, types[0]);
        Assert.Equal(1, types[15]);

        // A prefix that ends inside the token list or the tensor table holds no table; one that ends at the table's end does.
        Assert.Null(GgufHeaderProbe.ReadTensorTypes(bytes.AsSpan(0, 1024)));
        Assert.Null(GgufHeaderProbe.ReadTensorTypes(new byte[64]));
        int low = 0, high = bytes.Length;
        while (low < high)
        {
            int middle = (low + high) / 2;
            if (GgufHeaderProbe.ReadTensorTypes(bytes.AsSpan(0, middle)) is null)
            {
                low = middle + 1;
            }
            else
            {
                high = middle;
            }
        }

        Assert.True(low < bytes.Length - 300, "the table ends before the tensor data");
        Assert.Null(GgufHeaderProbe.ReadTensorTypes(bytes.AsSpan(0, low - 1)));
        Assert.NotEmpty(GgufHeaderProbe.ReadMetadata(bytes.AsSpan(0, low - 1)));
    }

    [Fact]
    public void ReadTensorTypes_RealFiles()
    {
        string quantized = Path.Combine(s_models, "quant-test/qwen3-0.6b-IQ2_S.gguf");
        string projector = Path.Combine(s_models, "mmproj-tiny-random-glm4v-f32.gguf");
        Assert.SkipUnless(File.Exists(quantized) && File.Exists(projector), "test GGUFs not found");

        var types = GgufHeaderProbe.ReadTensorTypes(Prefix(quantized, 32 << 20))!;
        Assert.NotNull(types);
        Assert.Contains(22, types.Keys); // IQ2_S
        Assert.All(types.Keys, id => Assert.NotNull(GgufSupport.GetTensorType(id)));
        using var file = GgufFile.Open(quantized);
        Assert.Equal(file.Tensors.Count, types.Values.Sum());

        var metadata = GgufHeaderProbe.ReadMetadata(Prefix(projector, 1 << 20));
        Assert.True(GgufSupport.IsProjector(metadata["general.architecture"] as string));
    }

    [Fact]
    public void Import_UnknownArchitecture_NamesIt()
    {
        string path = Path.Combine(s_models, "tiny-random-hrm-text-f32.gguf");
        Assert.SkipUnless(File.Exists(path), "test GGUF not found");
        var error = Assert.Throws<UnsupportedArchitectureException>(() => GgufImporter.Open(path));
        Assert.Equal("hrm_text", error.Architecture);
    }

    private static byte[] Prefix(string path, int count)
    {
        using var file = File.OpenRead(path);
        var buffer = new byte[(int)Math.Min(count, file.Length)];
        file.ReadExactly(buffer);
        return buffer;
    }
}
