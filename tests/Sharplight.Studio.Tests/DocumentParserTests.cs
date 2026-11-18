using Sharplight.Studio.Services.Search;

namespace Sharplight.Studio.Tests;

/// <summary>The Search page's document reading: one test per format over generated fixtures, and the size and zip bomb limits.</summary>
public sealed class DocumentParserTests : IDisposable
{
    private readonly TempDir _dir = new();
    private readonly StructuredDocumentParser _parser = new();

    public void Dispose() => _dir.Dispose();

    private IReadOnlyList<DocumentSection> Read(string name) => _parser.Parse(_dir.File(name), TestContext.Current.CancellationToken);

    [Fact]
    public void Markdown_IsCutAtHeadings_WithTheirFirstLine_AndFencedCodeIsNotAHeading()
    {
        File.WriteAllText(_dir.File("guide.md"), "# Guide\nIntro text.\n\n## Install\nRun the installer.\n```\n# not a heading\n```\n\n## Use\nStart it.\n");
        var sections = Read("guide.md");
        Assert.Equal(["Guide", "Install", "Use"], sections.Select(section => section.Heading));
        Assert.Equal([1, 4, 10], sections.Select(section => section.Line));
        Assert.Contains("# not a heading", sections[1].Text, StringComparison.Ordinal);
        Assert.All(sections, section => Assert.Equal(0, section.Page));
    }

    [Fact]
    public void Html_KeepsTheTextOnly_AndCutsAtHeadings()
    {
        File.WriteAllText(_dir.File("page.html"),
            "<html><head><title>t</title><style>p{color:red}</style><script>var secret=1;</script></head><body><!-- hidden -->"
            + "<h1>Tides</h1><p>The moon pulls the sea &amp; the shore.</p><h2>Waves</h2><p>Wind makes <b>waves</b>.</p></body></html>");
        var sections = Read("page.html");
        Assert.Equal(["Tides", "Waves"], sections.Select(section => section.Heading));
        Assert.Contains("the sea & the shore", sections[0].Text, StringComparison.Ordinal);
        Assert.Contains("Wind makes waves.", sections[1].Text, StringComparison.Ordinal);
        Assert.DoesNotContain(sections, section => section.Text.Contains("secret", StringComparison.Ordinal) || section.Text.Contains("color", StringComparison.Ordinal)
            || section.Text.Contains("hidden", StringComparison.Ordinal) || section.Text.Contains('<', StringComparison.Ordinal));
    }

    [Fact]
    public void Pdf_IsReadPageByPage()
    {
        DocumentFixtures.Pdf(_dir.File("book.pdf"), "Chapter one is about harbours.", "", "Chapter three covers lighthouses.");
        var sections = Read("book.pdf");
        Assert.Equal([1, 3], sections.Select(section => section.Page));
        Assert.Contains("harbours", sections[0].Text, StringComparison.Ordinal);
        Assert.Contains("lighthouses", sections[1].Text, StringComparison.Ordinal);
    }

    [Fact]
    public void Docx_IsCutAtHeadingParagraphs_AndTablesBecomeRows()
    {
        DocumentFixtures.Docx(_dir.File("report.docx"), "Preface paragraph.", "#Budget", "Money is tight.", "|Item|Cost|", "|Tea|3|", "#Plan", "Save more.");
        var sections = Read("report.docx");
        Assert.Equal(["", "Budget", "Plan"], sections.Select(section => section.Heading));
        Assert.Contains("| Tea | 3 |", sections[1].Text, StringComparison.Ordinal);
        Assert.Contains("Save more.", sections[2].Text, StringComparison.Ordinal);
    }

    [Fact]
    public void Pptx_IsReadSlideBySlide_HeadedByTheFirstLine()
    {
        DocumentFixtures.Pptx(_dir.File("deck.pptx"), "Welcome\nTo the quarterly review", "Results\nRevenue doubled");
        var sections = Read("deck.pptx");
        Assert.Equal([1, 2], sections.Select(section => section.Page));
        Assert.Equal(["Welcome", "Results"], sections.Select(section => section.Heading));
        Assert.Contains("Revenue doubled", sections[1].Text, StringComparison.Ordinal);
    }

    [Fact]
    public void Xlsx_IsReadSheetBySheet_WithSharedInlineAndNumberCells()
    {
        DocumentFixtures.Xlsx(_dir.File("sheet.xlsx"),
            ("Fruit", [["Apple", "#3", "~green"], ["Pear", "#5"]]),
            ("Empty", []),
            ("Cities", [["Oslo", "Norway"]]));
        var sections = Read("sheet.xlsx");
        Assert.Equal(["Fruit", "Cities"], sections.Select(section => section.Heading));
        Assert.Equal([1, 3], sections.Select(section => section.Page));
        Assert.Equal("Apple | 3 | green\nPear | 5", sections[0].Text.ReplaceLineEndings("\n"));
    }

    [Fact]
    public void SourceCode_AndPlainText_AreOneSectionFromLineOne()
    {
        File.WriteAllText(_dir.File("Program.cs"), "namespace Demo;\n\nclass Program\n{\n}\n");
        File.WriteAllText(_dir.File("notes.txt"), "just some notes");
        var code = Assert.Single(Read("Program.cs"));
        Assert.Equal(1, code.Line);
        Assert.Contains("class Program", code.Text, StringComparison.Ordinal);
        Assert.Equal("just some notes", Assert.Single(Read("notes.txt")).Text);
        Assert.True(_parser.CanParse("x.py"));
        Assert.True(_parser.CanParse("Dockerfile"));
        Assert.False(_parser.CanParse("photo.png"));
    }

    [Fact]
    public void FilesAboveTheSizeLimit_AreRefused()
    {
        var parser = new StructuredDocumentParser(new DocumentLimits { MaxTextBytes = 1000, MaxPackageBytes = 2000 });
        File.WriteAllText(_dir.File("big.txt"), new string('x', 1001));
        Assert.Equal(1000, parser.MaxBytes(_dir.File("big.txt")));
        Assert.Equal(2000, parser.MaxBytes(_dir.File("x.docx")));
        var error = Assert.Throws<InvalidDataException>(() => parser.Parse(_dir.File("big.txt"), TestContext.Current.CancellationToken));
        Assert.Contains("larger than", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AZipBomb_IsRefusedBeforeItIsUnpacked()
    {
        DocumentFixtures.ZipBomb(_dir.File("bomb.docx"), 20 << 20);   // 20 MB of zeros compress to about 20 KB
        Assert.True(new FileInfo(_dir.File("bomb.docx")).Length < 200_000);
        var ratio = new StructuredDocumentParser(new DocumentLimits { RatioFloorBytes = 1 << 20 });
        Assert.Contains("compressed", Assert.Throws<InvalidDataException>(() => ratio.Parse(_dir.File("bomb.docx"), TestContext.Current.CancellationToken)).Message, StringComparison.Ordinal);
        var total = new StructuredDocumentParser(new DocumentLimits { MaxUnpackedBytes = 1 << 20 });
        Assert.Contains("unpacks to more than", Assert.Throws<InvalidDataException>(() => total.Parse(_dir.File("bomb.docx"), TestContext.Current.CancellationToken)).Message, StringComparison.Ordinal);
        var entries = new StructuredDocumentParser(new DocumentLimits { MaxEntries = 0 });
        Assert.Throws<InvalidDataException>(() => entries.Parse(_dir.File("bomb.docx"), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void ADamagedOfficeFile_ReportsInvalidData()
    {
        File.WriteAllText(_dir.File("broken.docx"), "this is not a zip");
        Assert.Throws<InvalidDataException>(() => Read("broken.docx"));
    }

    [Fact]
    public void TheCharacterLimit_CutsALongDocument()
    {
        var parser = new StructuredDocumentParser(new DocumentLimits { MaxCharacters = 100 });
        File.WriteAllText(_dir.File("long.md"), "# One\n" + new string('a', 60) + "\n# Two\n" + new string('b', 60) + "\n");
        Assert.Equal(["One"], parser.Parse(_dir.File("long.md"), TestContext.Current.CancellationToken).Select(section => section.Heading));
    }
}
