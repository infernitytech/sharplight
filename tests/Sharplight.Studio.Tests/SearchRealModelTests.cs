using Sharplight.Studio.Models;
using Sharplight.Studio.ViewModels.Search;

namespace Sharplight.Studio.Tests;

/// <summary>The Search page with the real all-MiniLM-L6-v2 embedder over a generated corpus in every supported format (skipped when the model is missing).</summary>
public sealed class SearchRealModelTests
{
    private static readonly (string Query, string Expected)[] s_questions =
    [
        ("What do cats like to do?", "cats.md"),
        ("How long should I boil the pasta?", "recipes.docx"),
        ("When is the income tax return due?", "finance.xlsx"),
        ("Which planet is the largest?", "space.pdf"),
        ("How does the lighthouse warn ships?", "harbour.pptx"),
        ("How do I reverse a list in the program?", "Reverse.cs"),
        ("Why is the ocean salty?", "ocean.html"),
    ];

    private static void Corpus(string folder)
    {
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "cats.md"), "# Cats\nCats sleep for most of the day, purr when they are content and chase mice at night.\n\n# Dogs\nDogs bark at strangers and love long walks.\n");
        DocumentFixtures.Docx(Path.Combine(folder, "recipes.docx"), "#Pasta", "Boil the pasta in salted water for ten minutes, then drain it and add the tomato sauce.", "#Bread", "Knead the dough and let it rise for two hours.");
        DocumentFixtures.Xlsx(Path.Combine(folder, "finance.xlsx"), ("Deadlines", [["Income tax return", "due on the 15th of April"], ["Car insurance", "renew in June"]]));
        DocumentFixtures.Pdf(Path.Combine(folder, "space.pdf"), "Mercury is the smallest planet of the solar system.", "Jupiter is the largest planet and has a great red spot.");
        DocumentFixtures.Pptx(Path.Combine(folder, "harbour.pptx"), "Welcome\nAnnual meeting of the harbour board", "Lighthouse\nThe lighthouse beam and the foghorn warn ships of the rocks near the coast");
        File.WriteAllText(Path.Combine(folder, "Reverse.cs"), "static class Lists\n{\n    // Reverses the order of the items of a list in place.\n    public static void Reverse<T>(List<T> list) => list.Reverse();\n}\n");
        File.WriteAllText(Path.Combine(folder, "ocean.html"), "<html><body><h1>Oceans</h1><p>Rivers carry dissolved minerals into the sea, which is why the ocean is salty.</p></body></html>");
    }

    [Fact]
    public async Task AllMiniLm_FindsTheRightFileInEveryFormat_AndTheIndexSurvivesARestart()
    {
        string embed = RealStudioSession.Require("hf/all-MiniLM-L6-v2");
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        await session.LoadAsync("minilm", embed, ModelKindCatalog.Embeddings, ct);
        string folder = Path.Combine(session.Directory, "docs");
        Corpus(folder);

        var page = new SearchPageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        page.SetFolder(folder);
        page.IndexCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TimeSpan.FromMinutes(3), ct);
        Assert.True(page.HasIndex);
        Assert.StartsWith("7 documents, ", page.IndexSummary, StringComparison.Ordinal);

        foreach (var mode in new[] { SearchMode.Semantic, SearchMode.Hybrid })
        {
            page.Panel.Apply(new Dictionary<string, string> { ["mode"] = mode, ["topk"] = "3" });
            foreach (var (query, expected) in s_questions)
            {
                page.Query = query;
                page.SearchCommand.Execute(null);
                await page.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromMinutes(1), ct);
                Assert.True(expected == page.Results[0].FileName, $"{mode}: '{query}' -> {page.Results[0].FileName} ({page.Results[0].Location}), expected {expected}");
            }
        }

        // Keyword search finds an exact word without the model.
        page.Panel.Apply(new Dictionary<string, string> { ["mode"] = SearchMode.Keyword });
        page.Query = "foghorn";
        page.SearchCommand.Execute(null);
        await page.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromMinutes(1), ct);
        Assert.Equal("harbour.pptx", page.Results[0].FileName);
        Assert.Equal("slide 2 · Lighthouse", page.Results[0].Location);
        Assert.Contains("foghorn", page.Results[0].Segments.Single(segment => segment.IsMatch).Text, StringComparison.Ordinal);

        // A restart: the stored index loads for the folder and an update with no change embeds nothing.
        var reopened = new SearchPageViewModel(session.Services, new Fakes.FakeNavigator());
        await reopened.ActivateAsync();
        reopened.SetFolder(folder);
        await reopened.StoredIndexLoad;
        Assert.Equal(page.IndexSummary, reopened.IndexSummary);
        reopened.IndexCommand.Execute(null);
        await reopened.Queue.Items.Single().Completion.WaitAsync(TimeSpan.FromMinutes(1), ct);
        var update = reopened.History.Items[0].Record.Metadata;
        Assert.Equal(("0", "0", "0"), (update["added"], update["changed"], update["removed"]));

        // One file changes: only it is embedded again, and the new text is found.
        File.WriteAllText(Path.Combine(folder, "cats.md"), "# Cats\nCats climb trees and nap in the sun.\n\n# Volcanoes\nA volcano erupts when magma rises through the crust.\n");
        reopened.IndexCommand.Execute(null);
        await reopened.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromMinutes(1), ct);
        Assert.Equal("1", reopened.History.Items[0].Record.Metadata["changed"]);
        reopened.Query = "What makes a volcano erupt?";
        reopened.SearchCommand.Execute(null);
        await reopened.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromMinutes(1), ct);
        Assert.Equal("cats.md", reopened.Results[0].FileName);
        Assert.Equal("Volcanoes", reopened.Results[0].Hit.Heading);
    }
}
