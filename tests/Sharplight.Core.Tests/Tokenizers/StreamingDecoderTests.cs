using System.Text;
using Sharplight.Tokenizers.Bpe;
using Xunit;

namespace Sharplight.Tokenizers.Tests;

public class StreamingDecoderTests
{
    [Fact]
    public void BuffersPartialUtf8SequencesAcrossTokens()
    {
        var t = TokenizerFactory.Create(TestData.Tiny());
        // byte-level tokens: each byte of a multi-byte char is its own token here
        string text = "a€😀é";
        int[] ids = Encoding.UTF8.GetBytes(text).Select(b => t.TokenId(ByteEncoding.ByteToChar[b].ToString())!.Value).ToArray();
        var dec = new StreamingDecoder(t);
        var sb = new StringBuilder();
        var emitted = new List<string>();
        foreach (int id in ids)
        {
            string piece = dec.Push(id);
            emitted.Add(piece);
            Assert.DoesNotContain('�', piece);
            sb.Append(piece);
        }
        sb.Append(dec.Flush());
        Assert.Equal(text, sb.ToString());
        Assert.Equal("a", emitted[0]);
        Assert.Equal(["", "", "€"], emitted.Skip(1).Take(3));   // 3-byte char arrives on the third byte
        Assert.Equal(["", "", "", "😀"], emitted.Skip(4).Take(4));
        Assert.Equal(["", "é"], emitted.Skip(8).Take(2));
    }

    [Fact]
    public void FlushEmitsReplacementForDanglingBytes()
    {
        var t = TokenizerFactory.Create(TestData.Tiny());
        var dec = new StreamingDecoder(t);
        Assert.Equal("", dec.Push(t.TokenId(ByteEncoding.ByteToChar[0xE2].ToString())!.Value));
        Assert.Equal("�", dec.Flush());
        Assert.Equal("", dec.Flush());
    }

    [Fact]
    public void SkipSpecialDropsControlTokens()
    {
        var t = TokenizerFactory.Create(TestData.Tiny());
        var dec = new StreamingDecoder(t, skipSpecial: true);
        Assert.Equal("", dec.Push(t.TokenId("</s>")!.Value));
        Assert.Equal("hello", dec.Push(t.Encode("hello")));
    }
}
