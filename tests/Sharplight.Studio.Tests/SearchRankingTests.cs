using Sharplight.Studio.Services.Search;

namespace Sharplight.Studio.Tests;

/// <summary>BM25, reciprocal rank fusion, highlighting and the location labels of the Search page.</summary>
public sealed class SearchRankingTests
{
    private static readonly string[] s_passages =
    [
        "The cat sat on the mat.",
        "Dogs bark at the mailman every morning.",
        "A cat and another cat chase a third cat across the garden.",
        "Error code E1042 means the disk is full.",
        "Quarterly tax returns are due in April.",
    ];

    [Fact]
    public void Tokenize_LowercasesLettersAndDigits()
    {
        Assert.Equal(["error", "e1042", "disk", "full"], Bm25Index.Tokenize("Error: E1042 (disk-full)!"));
        Assert.Empty(Bm25Index.Tokenize("  ...  "));
    }

    [Fact]
    public void Bm25_RanksByRareWordsAndFrequency_AndLeavesOutPassagesWithoutQueryWords()
    {
        var index = new Bm25Index(s_passages);
        var cat = index.Search("cat", 10);
        Assert.Equal([2, 0], cat.Select(hit => hit.Index));   // three mentions beat one, even in a longer passage
        Assert.True(cat[0].Score > cat[1].Score);

        var code = index.Search("what does E1042 mean", 10);
        Assert.Equal(3, code[0].Index);   // the rare token decides, "what", "does" and "mean" appear nowhere else or once
        Assert.Empty(index.Search("zebra", 10));
        Assert.Single(index.Search("cat", 1));
        Assert.Empty(new Bm25Index([]).Search("cat", 5));
    }

    [Fact]
    public void Bm25_ScoresMatchTheFormula()
    {
        var index = new Bm25Index(["alpha beta", "beta gamma gamma", "delta"]);
        // N = 3, n(alpha) = 1: idf = ln(1 + 2.5 / 1.5); avg length = 2, passage 0 has length 2: tf = 1, norm = 1.
        double expected = Math.Log(1 + 2.5 / 1.5) * 1 * 2.2 / (1 + 1.2 * 1);
        Assert.Equal(expected, index.Search("alpha", 1)[0].Score, 9);
    }

    [Fact]
    public void Fusion_SumsReciprocalRanks()
    {
        // Ranking A: 10, 20, 30. Ranking B: 30, 10. k = 60.
        var fused = RankFusion.Fuse([[10, 20, 30], [30, 10]]);
        Assert.Equal([10, 30, 20], fused.Select(hit => hit.Id));
        Assert.Equal(1.0 / 61 + 1.0 / 62, fused[0].Score, 12);   // 10: rank 1 and rank 2
        Assert.Equal(1.0 / 63 + 1.0 / 61, fused[1].Score, 12);   // 30: rank 3 and rank 1
        Assert.Equal(1.0 / 62, fused[2].Score, 12);
    }

    [Fact]
    public void Fusion_RewardsItemsBothRankingsLike_AndBreaksTiesByBestRankThenId()
    {
        // 1 is second in both rankings; 2 and 3 are first in only one each: 1/61 + ... : 2/62 > 1/61.
        var fused = RankFusion.Fuse([[2, 1], [3, 1]]);
        Assert.Equal(1, fused[0].Id);
        Assert.Equal([2, 3], fused.Skip(1).Select(hit => hit.Id));   // equal scores: both best rank 0, lower id first
        Assert.Empty(RankFusion.Fuse([]));
        Assert.Equal([5], RankFusion.Fuse([[5], []]).Select(hit => hit.Id));
        Assert.Equal(1.0 / 6, RankFusion.Fuse([[5]], k: 5)[0].Score, 12);
    }

    [Fact]
    public void Highlighter_MarksQueryWords_IgnoringCaseAndStopWords()
    {
        var segments = QueryHighlighter.Split("The Cat sat on the mat. Cats too.", "where is the cat on the mat");
        Assert.Equal("The Cat sat on the mat. Cats too.", string.Concat(segments.Select(segment => segment.Text)));
        Assert.Equal(["Cat", "mat"], segments.Where(segment => segment.IsMatch).Select(segment => segment.Text));
        Assert.Single(QueryHighlighter.Split("nothing here", "zebra"));
        Assert.False(Assert.Single(QueryHighlighter.Split("the of a", "the of a")).IsMatch);   // only stop words: nothing to highlight
    }

    [Theory]
    [InlineData("a/x.pdf", 3, 0, "", "page 3")]
    [InlineData("a/x.pptx", 2, 0, "Results", "slide 2 · Results")]
    [InlineData("a/x.xlsx", 1, 0, "Fruit", "sheet 1 · Fruit")]
    [InlineData("a/x.cs", 0, 41, "", "line 41")]
    [InlineData("a/x.md", 0, 7, "Install", "line 7 · Install")]
    [InlineData("a/x.docx", 0, 0, "Budget", "Budget")]
    [InlineData("a/x.txt", 0, 0, "", "")]
    public void Locations_AreNamedInTheFormatsWords(string file, int page, int line, string heading, string expected)
    {
        Assert.Equal(expected, LocationLabel.For(file, page, line, heading));
    }

    [Fact]
    public void Opener_UsesPositionAwareProgramsOnlyWhenInstalled()
    {
        string? Installed(string name) => name is "code" or "okular" ? "/usr/bin/" + name : null;
        string? None(string name) => null;

        var line = LocationOpener.Command("/d/a.cs", 0, 41, Installed);
        Assert.Equal("/usr/bin/code", line!.Value.Program);
        Assert.Equal(["--goto", "/d/a.cs:41"], line.Value.Arguments);
        var page = LocationOpener.Command("/d/a.pdf", 7, 0, Installed);
        Assert.Equal("/usr/bin/okular", page!.Value.Program);
        Assert.Equal(["--page", "7", "/d/a.pdf"], page.Value.Arguments);
        Assert.Null(LocationOpener.Command("/d/a.pdf", 7, 0, None));      // no known viewer: the default app opens it
        Assert.Null(LocationOpener.Command("/d/a.cs", 0, 41, None));
        Assert.Null(LocationOpener.Command("/d/a.docx", 0, 3, Installed));   // an editor cannot place a line in a Word file
        Assert.Null(LocationOpener.Command("/d/a.cs", 0, 0, Installed));     // no position: nothing to do
    }

    [Fact]
    public void Citations_AreReadInOrderOfFirstMention_WithinTheGivenPassages()
    {
        Assert.Equal([2, 1], AnswerPrompt.Citations("It is in [2]. Also see [1][2] and [9] and [0].", 3));
        Assert.Empty(AnswerPrompt.Citations("No sources.", 3));
    }
}
