using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Gguf.Tests;

public class RealModelTests
{
    // Optional local models (the tests return early when a file is absent), relative to the user's home directory.
    private static readonly string Q8 = Home("models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf");
    private const string Q4KM = ".lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf";
    private const string Nvfp4 = ".lmstudio/models/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-MTP-ORIG.gguf";

    private static string Home(string relative) => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), relative);

    [Fact]
    public void Qwen_Q8_0()
    {
        if (!File.Exists(Q8))
        {
            return;
        }

        using var f = GgufFile.Open(Q8);
        Assert.Equal(3u, f.Version);
        Assert.Equal(866, f.Tensors.Count);
        Assert.Equal(39, f.Metadata.Count);
        Assert.Equal("qwen35", f.Architecture);
        Assert.Equal(65, f.GetArch<int>("block_count"));
        var t = f.GetTensor("token_embd.weight");
        Assert.Equal(DataType.Q8_0, t.DataType);
        Assert.Equal(new long[] { 5120, 248320 }, t.Ne);
        Assert.Equal(new Shape(248320, 5120), t.Shape);
        Assert.Equal(248320L * 5120 / 32 * 34, t.ByteSize);
        Assert.Equal(248320, f.GetArray<string>("tokenizer.ggml.tokens").Length);
        // read only a small slice of a big tensor
        var w = t.ToWeightInfo();
        var buf = new byte[34 * 4];
        w.Source.Read(w.Source.ByteLength - buf.Length, buf);
        Assert.NotEqual(0, w.Source.TryGetPointer());
    }

    [Theory]
    [InlineData(Q4KM)]
    [InlineData(Nvfp4)]
    public void Other_Models_Parse(string relative)
    {
        string path = Home(relative);
        if (!File.Exists(path))
        {
            return;
        }

        using var f = GgufFile.Open(path);
        Assert.NotEmpty(f.Tensors);
        Assert.NotNull(f.Architecture);
        var hist = string.Join(", ", f.Tensors.GroupBy(t => t.DataType).Select(g => $"{g.Key}={g.Count()}"));
        Console.WriteLine($"{Path.GetFileName(path)}: {hist}");
    }
}
