using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Gguf.Tests;

/// <summary><see cref="GgufHeaderProbe"/> reads the dimensions from the start of a file, as a range request before a download fetches it.</summary>
public class GgufHeaderProbeTests
{
    private static byte[] Write()
    {
        var tokens = Enumerable.Range(0, 5000).Select(i => $"token-{i}").ToArray();
        using var stream = new MemoryStream();
        new GgufWriter()
            .AddMetadata("general.architecture", "llama")
            .AddMetadata("llama.block_count", 32u)
            .AddMetadata("llama.attention.head_count", 32u)
            .AddMetadata("llama.attention.head_count_kv", new[] { 8u, 4u, 8u })
            .AddMetadata("llama.embedding_length", 4096u)
            .AddMetadata("llama.context_length", 131072u)
            .AddMetadata("tokenizer.ggml.tokens", tokens)
            .AddTensor("w", DataType.F32, new byte[16], 4)
            .Write(stream);
        return stream.ToArray();
    }

    [Fact]
    public void ReadShape_WholeHeader_HasDimensionsAndVocabulary()
    {
        var shape = GgufHeaderProbe.ReadShape(Write());
        Assert.Equal("llama", shape.Architecture);
        Assert.Equal(32, shape.BlockCount);
        Assert.Equal(32, shape.HeadCount);
        Assert.Equal(8, shape.KvHeadCount);
        Assert.Equal(4096, shape.EmbeddingLength);
        Assert.Equal(131072, shape.ContextLength);
        Assert.Equal(5000, shape.VocabularySize);
        Assert.Null(shape.KeyLength);
    }

    [Fact]
    public void ReadShape_PrefixEndingInTheTokenList_KeepsTheDimensions()
    {
        var bytes = Write();
        var shape = GgufHeaderProbe.ReadShape(bytes.AsSpan(0, 1024));
        Assert.Equal(32, shape.BlockCount);
        Assert.Equal(8, shape.KvHeadCount);
        Assert.Null(shape.VocabularySize);
    }

    [Fact]
    public void ReadMetadata_TooShortOrNotGguf_IsEmpty()
    {
        Assert.Empty(GgufHeaderProbe.ReadMetadata(Write().AsSpan(0, 10)));
        Assert.Empty(GgufHeaderProbe.ReadMetadata(new byte[64]));
        Assert.Null(GgufHeaderProbe.ReadShape(new byte[64]).BlockCount);
    }
}
