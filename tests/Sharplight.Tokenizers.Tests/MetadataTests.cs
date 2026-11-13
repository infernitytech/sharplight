using Xunit;

namespace Sharplight.Tokenizers.Tests;

public class MetadataTests
{
    [Fact]
    public void FromMetadata_IsLenientAboutNumberAndArrayTypes()
    {
        var md = new Dictionary<string, object>
        {
            ["tokenizer.model"] = "gpt2",
            ["tokenizer.pre"] = "qwen35",
            ["tokenizer.tokens"] = new object[] { "a", "b", "<x>" },
            ["tokenizer.token_type"] = new uint[] { 1, 1, 3 },
            ["tokenizer.merges"] = new List<string> { "a b" },
            ["tokenizer.bos_token_id"] = 2UL,
            ["tokenizer.eos_token_id"] = (uint)1,
            ["tokenizer.eot_token_id"] = 2L,
            ["tokenizer.eom_token_id"] = (byte)1,
            ["tokenizer.padding_token_id"] = 0,
            ["tokenizer.add_bos_token"] = false,
            ["tokenizer.add_eos_token"] = (byte)1,
        };
        var d = TokenizerData.FromMetadata(md);
        Assert.Equal("gpt2", d.Model);
        Assert.Equal("qwen35", d.PreTokenizer);
        Assert.Equal(["a", "b", "<x>"], d.Tokens);
        Assert.Equal([1, 1, 3], d.TokenTypes);
        Assert.Equal(["a b"], d.Merges);
        Assert.Equal(2, d.BosId);
        Assert.Equal(1, d.EosId);
        Assert.Equal([1, 2], d.EosIds);
        Assert.Equal(0, d.PadId);
        Assert.False(d.AddBos);
        Assert.True(d.AddEos);
    }

    [Fact]
    public void FromMetadata_MissingOptionalKeys()
    {
        var d = TokenizerData.FromMetadata(new Dictionary<string, object> { ["tokenizer.tokens"] = new[] { "a" } });
        Assert.Null(d.BosId);
        Assert.Null(d.AddBos);
        Assert.Empty(d.EosIds);
        Assert.Null(d.Merges);
    }

    [Fact]
    public void Factory_RejectsUnknownModel()
        => Assert.Throws<NotSupportedException>(() => TokenizerFactory.Create(new TokenizerData { Model = "no-such-model", Tokens = [] }));
}
