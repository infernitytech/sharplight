using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

public class StopStringFilterTests
{
    private static (string Out, string? Matched) Run(string[] stops, params string[] pieces)
    {
        var f = new StopStringFilter(stops);
        var o = new System.Text.StringBuilder();
        foreach (var p in pieces) { o.Append(f.Push(p)); if (f.Stopped)
            {
                break;
            }
        }
        if (!f.Stopped)
        {
            o.Append(f.Flush());
        }

        return (o.ToString(), f.Matched);
    }

    [Fact] public void Match_InsideOnePiece() => Assert.Equal(("ab", "XY"), Run(["XY"], "abXYcd"));
    [Fact] public void Match_AcrossPieces() => Assert.Equal(("ab", "XYZ"), Run(["XYZ"], "abX", "Y", "Zcd"));
    [Fact] public void NoMatch_ReleasesHeldText() => Assert.Equal(("abXYcd", null), Run(["XYZ"], "abX", "Y", "cd"));
    [Fact] public void TrailingPartial_FlushedAtEnd() => Assert.Equal(("abXY", null), Run(["XYZ"], "ab", "XY"));
    [Fact] public void FirstOfSeveral_Wins() => Assert.Equal(("a", "bb"), Run(["cc", "bb"], "a", "b", "bcc"));
    [Fact] public void Overlapping_PartialRestarts() => Assert.Equal(("aab", "abc"), Run(["abc"], "aa", "b", "abc"));   // "aab" then "abc"
    [Fact] public void Empty_StopStrings_AreIgnored() => Assert.Equal(("abc", null), Run([""], "a", "bc"));
}
