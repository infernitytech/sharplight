using Sharplight.CatalogTool;
using Sharplight.Hub.Catalog;

namespace Sharplight.Hub.Tests;

public sealed class VerifiedCatalogTests
{
    private static readonly string s_root = CatalogPaths.FindRepositoryRoot(AppContext.BaseDirectory);

    private static VerifiedCatalog Committed() => VerifiedCatalogSerializer.Parse(File.ReadAllText(Path.Combine(s_root, CatalogPaths.Catalog)));

    [Fact]
    public void CommittedCatalogIsValid()
    {
        var catalog = Committed();

        Assert.Empty(CatalogValidator.Validate(catalog));
        Assert.Contains(catalog.Entries, entry => entry.IsInstallable && entry.Kind == "text-generation");
        Assert.All(catalog.Entries.Where(entry => entry.IsInstallable), entry => Assert.NotNull(entry.DefaultVariant));
    }

    [Fact]
    public void CommittedCatalogCoversEveryRunnableTask()
    {
        var kinds = Committed().Entries.Where(entry => entry.IsInstallable).Select(entry => entry.Kind).ToHashSet();

        Assert.Subset(kinds, new HashSet<string> { "text-generation", "speech-recognition", "speech-synthesis", "image-generation", "video-generation", "embeddings", "ocr" });
    }

    [Fact]
    public void SchemaFileMatchesTheTypes()
    {
        var committed = File.ReadAllText(Path.Combine(s_root, "catalog", CatalogPaths.SchemaFileName)).ReplaceLineEndings("\n");

        Assert.Equal(VerifiedCatalogSerializer.ExportSchema(), committed);
    }

    [Fact]
    public void CompatibleModelsTablesAreGeneratedFromTheCatalog()
    {
        var document = File.ReadAllText(Path.Combine(s_root, CatalogPaths.Document)).ReplaceLineEndings("\n");

        Assert.Equal(document, CompatibleModelsDocument.Render(document, Committed()));
    }

    [Fact]
    public void SerializerRoundTrips()
    {
        var catalog = Committed();

        var again = VerifiedCatalogSerializer.Parse(VerifiedCatalogSerializer.Serialize(catalog));

        Assert.Equal(VerifiedCatalogSerializer.Serialize(catalog), VerifiedCatalogSerializer.Serialize(again));
    }

    [Fact]
    public void NewerSchemaVersionIsRejected()
    {
        Assert.Throws<InvalidDataException>(() => VerifiedCatalogSerializer.Parse("""{"schemaVersion": 2, "entries": []}"""));
        Assert.Throws<InvalidDataException>(() => VerifiedCatalogSerializer.Parse("not json"));
    }

    [Fact]
    public void ValidatorReportsEachProblem()
    {
        var bad = CatalogSamples.Entry("bad", CatalogSamples.Variant(revision: "main", companions: "missing")) with { Kind = "nonsense" };
        var duplicate = CatalogSamples.Entry("dup");
        var noFiles = CatalogSamples.Entry("no-files", CatalogSamples.Variant() with { Files = [], Primary = "" });
        var catalog = CatalogSamples.Catalog(bad, duplicate, duplicate, noFiles);

        var problems = CatalogValidator.Validate(catalog);

        Assert.Contains(problems, problem => problem.Contains("unknown kind 'nonsense'", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("revision 'main'", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("unknown companion 'missing'", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("dup is used twice", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.StartsWith("no-files: variant q8-0-gguf: no files", StringComparison.Ordinal));
    }

    [Fact]
    public void CompanionsAreResolvedPerVariant()
    {
        var companion = new CatalogCompanion
        {
            Id = "vae", Role = "vae", Label = "VAE", Repository = "owner/Vae", Revision = CatalogSamples.Commit, Primary = "ae.safetensors",
            Files = [CatalogSamples.File("ae.safetensors")],
        };
        var entry = CatalogSamples.Entry("with-vae", CatalogSamples.Variant(companions: "vae")) with { Companions = [companion] };

        Assert.Empty(CatalogValidator.Validate(CatalogSamples.Catalog(entry)));
        Assert.Same(companion, Assert.Single(entry.CompanionsOf(entry.Variants[0])));
    }

    [Fact]
    public void DocumentTablesKeepTheProseAroundThem()
    {
        var catalog = CatalogSamples.Catalog(CatalogSamples.Entry() with { Doc = [new CatalogDocRow { Section = "t", Cells = new Dictionary<string, string> { ["notes"] = "a|b" } }] }) with
        {
            DocSections = [new CatalogDocSection { Id = "t", Columns = [new() { Key = "model", Title = "Model" }, new() { Key = "notes", Title = "Notes" }, new() { Key = "date", Title = "Verified" }] }],
        };
        var document = $"# Title\n\nProse.\n\n{CompatibleModelsDocument.BeginPrefix}t -->\n| old |\n{CompatibleModelsDocument.EndMarker}\n\nMore prose.\n";

        var rendered = CompatibleModelsDocument.Render(document, catalog);

        Assert.Equal($"# Title\n\nProse.\n\n{CompatibleModelsDocument.BeginPrefix}t -->\n| Model | Notes | Verified |\n|---|---|---|\n| Tiny model | a\\|b | 2026-11-16 |\n{CompatibleModelsDocument.EndMarker}\n\nMore prose.\n", rendered);
    }
}
