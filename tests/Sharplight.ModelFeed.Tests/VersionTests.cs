using Sharplight.ModelFeed.Versioning;

namespace Sharplight.ModelFeed.Tests;

public sealed class VersionTests
{
    [Theory]
    [InlineData("1.0.0", "1.0.1")]
    [InlineData("1.0.0-alpha", "1.0.0")]
    [InlineData("1.0.0-alpha", "1.0.0-alpha.1")]
    [InlineData("1.0.0-alpha.1", "1.0.0-alpha.beta")]
    [InlineData("1.0.0-beta.2", "1.0.0-beta.11")]
    [InlineData("1.9.0", "1.10.0")]
    public void Precedence_FollowsSemVer(string lower, string higher)
    {
        Assert.True(SemanticVersion.Parse(lower) < SemanticVersion.Parse(higher));
    }

    [Theory]
    [InlineData("1.0")]
    [InlineData("01.0.0")]
    [InlineData("1.0.0-")]
    [InlineData("1.0.0-01")]
    [InlineData("v1.0.0")]
    public void Parse_RejectsInvalid(string text)
    {
        Assert.False(SemanticVersion.TryParse(text, out _));
    }

    [Fact]
    public void BuildMetadata_IsIgnoredForPrecedence()
    {
        Assert.Equal(SemanticVersion.Parse("1.2.3+a"), SemanticVersion.Parse("1.2.3+b"));
    }

    [Theory]
    [InlineData("[1.0.0,2.0.0)", "1.0.0", true)]
    [InlineData("[1.0.0,2.0.0)", "2.0.0", false)]
    [InlineData("(1.0.0,2.0.0]", "1.0.0", false)]
    [InlineData("(1.0.0,2.0.0]", "2.0.0", true)]
    [InlineData("1.2.0", "3.0.0", true)]
    [InlineData("1.2.0", "1.1.9", false)]
    [InlineData("[1.2.0]", "1.2.0", true)]
    [InlineData("[1.2.0]", "1.2.1", false)]
    [InlineData("(,2.0.0)", "1.9.9", true)]
    [InlineData(">=1.0.0", "1.0.0", true)]
    [InlineData(">=1.0.0 <2.0.0", "2.0.0", false)]
    [InlineData(">1.0.0", "1.0.0", false)]
    [InlineData("=1.0.0", "1.0.0", true)]
    [InlineData("*", "9.9.9", true)]
    [InlineData("[1.0.0,2.0.0)", "1.5.0-rc.1", false)]
    [InlineData("[1.5.0-rc.1,2.0.0)", "1.5.0-rc.2", true)]
    public void Range_Satisfies(string range, string version, bool expected)
    {
        Assert.Equal(expected, VersionRange.Parse(range).Satisfies(SemanticVersion.Parse(version)));
    }

    [Theory]
    [InlineData("[2.0.0,1.0.0]")]
    [InlineData("[,]")]
    [InlineData("~1.0")]
    [InlineData("(1.0.0)")]
    public void Range_RejectsInvalid(string text)
    {
        Assert.False(VersionRange.TryParse(text, out _));
    }
}
