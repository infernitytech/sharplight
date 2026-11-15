using Sharplight.Tokenizers.Bpe;
using Xunit;

namespace Sharplight.Tokenizers.Tests;

public class ByteEncodingTests
{
    [Fact]
    public void MapIsBijectiveAndMatchesGpt2()
    {
        Assert.Equal('Ġ', ByteEncoding.ByteToChar[' ']);
        Assert.Equal('Ċ', ByteEncoding.ByteToChar['\n']);
        Assert.Equal('A', ByteEncoding.ByteToChar['A']);
        Assert.Equal(256, ByteEncoding.ByteToChar.Distinct().Count());
        for (int b = 0; b < 256; b++)
        {
            Assert.True(ByteEncoding.TryGetByte(ByteEncoding.ByteToChar[b], out byte back));
            Assert.Equal((byte)b, back);
        }
    }
}
