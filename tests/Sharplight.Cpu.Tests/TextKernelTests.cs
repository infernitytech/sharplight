using System.Text;

namespace Sharplight.Cpu.Tests;

/// <summary>The host text kernels' string algorithms: MurmurHash3 (x86, 32 bit) and Python-style string splitting.</summary>
public sealed class TextKernelTests
{
    [Theory]
    [InlineData("", 0u, 0u)]
    [InlineData("", 1u, 0x514e28b7u)]
    [InlineData("hello", 0u, 0x248bfa47u)]
    [InlineData("Hello, world!", 1234u, 0xfaf6cdb3u)]
    [InlineData("The quick brown fox jumps over the lazy dog", 0x9747b28cu, 0x2fa826cdu)]
    public void MurmurHash3_MatchesTheReferenceVectors(string text, uint seed, uint expected)
        => Assert.Equal(expected, MurmurHash3Op.Hash(Encoding.UTF8.GetBytes(text), seed));

    [Theory]
    [InlineData("a,b,,c", ",", -1, new[] { "a", "b", "", "c" })]
    [InlineData("a,b,c", ",", 1, new[] { "a", "b,c" })]
    [InlineData("", ",", -1, new[] { "" })]
    [InlineData("  hello   world ", "", -1, new[] { "hello", "world" })]
    [InlineData("  hello   big world ", "", 1, new[] { "hello", "big world " })]
    [InlineData("   ", "", -1, new string[0])]
    [InlineData("a<>b<>", "<>", -1, new[] { "a", "b", "" })]
    public void Split_FollowsPythonStrSplit(string text, string delimiter, long maxSplit, string[] expected)
        => Assert.Equal(expected, StringSplitOp.Split(text, delimiter, maxSplit));
}
