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
    public void LicenseNoticeIsOptionalAndRoundTrips()
    {
        var plain = CatalogSamples.Entry("plain");
        var noticed = CatalogSamples.Entry("noticed") with { License = "cc-by-nc-4.0", LicenseNotice = "Non-commercial use only." };

        var json = VerifiedCatalogSerializer.Serialize(CatalogSamples.Catalog(plain, noticed));
        var again = VerifiedCatalogSerializer.Parse(json);

        Assert.Equal(1, json.Split("\"licenseNotice\"").Length - 1);
        Assert.Null(again.Entries[0].LicenseNotice);
        Assert.Equal("Non-commercial use only.", again.Entries[1].LicenseNotice);
    }

    [Fact]
    public void OlderReadersIgnoreLicenseNotice()
    {
        // An older Studio deserializes the catalog with entry types that lack the field: unknown members must not break it, at any schema version they read.
        var older = System.Text.Json.JsonSerializer.Deserialize<OlderCatalog>(File.ReadAllText(Path.Combine(s_root, CatalogPaths.Catalog)),
            new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        var current = Committed();

        Assert.NotNull(older);
        Assert.Equal(current.Entries.Count, older.Entries.Count);
        Assert.Contains(current.Entries, entry => entry.LicenseNotice is not null);
        Assert.Equal(current.Entries.Select(entry => entry.License), older.Entries.Select(entry => entry.License));
        Assert.Equal(1, older.SchemaVersion);

        // The schema stays open for members it does not know, so editors and validators of older checkouts accept the new field.
        Assert.DoesNotContain("\"additionalProperties\": false", File.ReadAllText(Path.Combine(s_root, "catalog", CatalogPaths.SchemaFileName)), StringComparison.Ordinal);
        Assert.Contains("\"licenseNotice\"", VerifiedCatalogSerializer.ExportSchema(), StringComparison.Ordinal);
    }

    private sealed record OlderCatalog(int SchemaVersion, List<OlderEntry> Entries);

    private sealed record OlderEntry(string Id, string Name, string Kind, string? License, string? Notes);

    [Fact]
    public void ValidatorRequiresALicenseNoticeForNonPermissiveLicences()
    {
        var missing = CatalogSamples.Entry("missing") with { License = "other" };
        var blank = CatalogSamples.Entry("blank") with { License = "mit", LicenseNotice = "  " };
        var tooLong = CatalogSamples.Entry("long") with { License = "other", LicenseNotice = new string('x', CatalogValidator.MaxLicenseNoticeLength + 1) };
        var fine = CatalogSamples.Entry("fine") with { License = "other", LicenseNotice = "Read the terms." };
        var permissive = CatalogSamples.Entry("permissive") with { License = "apache-2.0" };
        var unknown = CatalogSamples.Entry("unknown") with { License = null };

        var problems = CatalogValidator.Validate(CatalogSamples.Catalog(missing, blank, tooLong, fine, permissive, unknown));

        Assert.Contains(problems, problem => problem.StartsWith("missing:", StringComparison.Ordinal) && problem.Contains("add a licenseNotice", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.StartsWith("blank:", StringComparison.Ordinal) && problem.Contains("licenseNotice must be", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.StartsWith("long:", StringComparison.Ordinal));
        Assert.DoesNotContain(problems, problem => problem.StartsWith("fine:", StringComparison.Ordinal) || problem.StartsWith("permissive:", StringComparison.Ordinal) || problem.StartsWith("unknown:", StringComparison.Ordinal));
    }

    [Fact]
    public void CommittedNoticesNameTheirLicence_NotTheGeneralNotes()
    {
        var catalog = Committed();

        foreach (var id in new[] { "omnivoice", "xtts-v2", "depth-anything-3-large", "gemma-3-4b-it", "flux.1-dev", "nemotron-3-nano-4b" })
        {
            var entry = catalog.Entries.Single(candidate => candidate.Id == id);
            Assert.False(string.IsNullOrWhiteSpace(entry.LicenseNotice), id);
        }

        Assert.DoesNotContain("Mamba", catalog.Entries.Single(entry => entry.Id == "nemotron-3-nano-4b").LicenseNotice, StringComparison.Ordinal);
        Assert.Contains("non-commercial", catalog.Entries.Single(entry => entry.Id == "xtts-v2").LicenseNotice, StringComparison.OrdinalIgnoreCase);
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
        var sameLabel = CatalogSamples.Entry("same-label", CatalogSamples.Variant(), CatalogSamples.Variant() with { Id = "q4-k-m-gguf" });
        var catalog = CatalogSamples.Catalog(bad, duplicate, duplicate, noFiles, sameLabel);

        var problems = CatalogValidator.Validate(catalog);

        Assert.Contains(problems, problem => problem.Contains("unknown kind 'nonsense'", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("revision 'main'", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("unknown companion 'missing'", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("dup is used twice", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.StartsWith("no-files: variant q8-0-gguf: no files", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.StartsWith("same-label: variant q4-k-m-gguf: the label", StringComparison.Ordinal) && problem.Contains("used twice", StringComparison.Ordinal));
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
