using System.Text.Json;
using Sharplight.CatalogTool;
using Sharplight.FeedTool;
using Sharplight.Hub.Catalog;
using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Publishing;
using Sharplight.ModelFeed.Signing;

namespace Sharplight.Hub.Tests;

public sealed class FeedToolTests
{
    private static readonly string s_root = CatalogPaths.FindRepositoryRoot(AppContext.BaseDirectory);

    private static VerifiedCatalog Catalog() => VerifiedCatalogSerializer.Parse(File.ReadAllText(Path.Combine(s_root, CatalogPaths.Catalog)));

    private static FeedOverlay Overlay() =>
        JsonSerializer.Deserialize(File.ReadAllBytes(Path.Combine(s_root, FeedToolPaths.Overlay)), FeedToolJsonContext.Default.FeedOverlay)!;

    [Fact]
    public void Overlay_NamesOnlyCatalogPackages()
    {
        var catalog = Catalog();
        var ids = catalog.Entries.Where(entry => entry.IsInstallable)
            .SelectMany(entry => entry.Variants.Select(variant => ManifestFactory.PackageId(entry, variant))).ToHashSet(StringComparer.Ordinal);
        var overlay = Overlay();
        Assert.All(overlay.Packages.Keys, id => Assert.Contains(id, ids));
        Assert.All(overlay.Packages.Values.SelectMany(package => package.Dependencies), dependency => Assert.Contains(dependency.Id, ids));
    }

    [Fact]
    public void Manifest_ComesFromCatalogEntryAndOverlay()
    {
        var entry = Catalog().Entries.Single(candidate => candidate.Id == "whisper-tiny");
        var variant = entry.Variants.Single(candidate => candidate.Id == "q5-1-whispercpp");
        var plan = CatalogInstallPlan.For(entry, variant);
        var file = new PackageFile { Name = "whisper-tiny.q5-1-whispercpp.1.0.0.slm", Role = PackageFileRoles.Package, Size = 1000, Sha256 = new string('a', 64) };
        var inspection = new PackageInspection("Whisper tiny", "speech-recognition", "whisper", 448, 39_000_000, 30_000_000, 3, false);
        var manifest = ManifestFactory.Create(plan, inspection, file, "1.0.0", Overlay().Packages["whisper-tiny.q5-1-whispercpp"], "0.4.2-alpha", null, DateTimeOffset.UnixEpoch);

        Assert.Equal("whisper-tiny.q5-1-whispercpp", manifest.Id);
        Assert.Equal([PackageTasks.Transcription], manifest.Model!.Tasks);
        Assert.Equal("OpenAI", manifest.Publisher?.Name);
        Assert.Equal("US", manifest.Publisher?.CountryOfOrigin);
        Assert.Equal("mit", manifest.License?.Spdx);
        Assert.True(manifest.License?.CommercialUse);
        Assert.Equal(">=0.4.0", manifest.Requirements!.Engine);
        Assert.Equal(variant.Revision, manifest.Provenance!.Sources[0].Revision);
        Assert.Equal(variant.Files[0].Sha256, manifest.Provenance.Sources[0].Files[0].Sha256);
        var dependency = Assert.Single(manifest.Dependencies);
        Assert.Equal("silero-vad-v5.onnx", dependency.Id);
        Assert.True(dependency.IsRequired);
        Assert.Equal("verified", manifest.Verification!.Level);

        // The manifest round-trips through the feed reader.
        var parsed = FeedJson.ReadManifest(FeedJson.Write(manifest, FeedJsonContext.Default.PackageManifest));
        Assert.Equal(manifest.Id, parsed.Id);
    }

    [Theory]
    [InlineData("qwen3-0.6b", "chat")]
    [InlineData("all-minilm-l6-v2", "embedding")]
    [InlineData("ms-marco-minilm-l6-v2", "rerank")]
    [InlineData("silero-vad-v5", "voice-activity")]
    public void Tasks_FollowKindAndTags(string entryId, string task)
    {
        var entry = Catalog().Entries.Single(candidate => candidate.Id == entryId);
        Assert.Contains(task, ManifestFactory.Describe(entry).Tasks);
    }

    [Fact]
    public async Task Versions_FollowPinsAndContent()
    {
        using var directory = new TempDirectory();
        var feed = directory.Combine("feed");
        Directory.CreateDirectory(feed);
        using var root = FeedSigningKey.Generate();
        using var key = FeedSigningKey.Generate();
        FeedPublisher.WriteKeySet(feed, new FeedKeySet { Version = 1, Issued = DateTimeOffset.UtcNow.AddDays(-1), Keys = [new FeedKeySetEntry { KeyId = key.KeyId, PublicKey = key.PublicKey.Base64 }] }, root);
        ProvenanceSource[] pins = [new() { Repository = "org/model", Revision = new string('1', 40), Role = "primary" }];
        Assert.Equal("1.0.0", VersionPlanner.Next(feed, "model.q8", pins, new string('a', 64)));

        var staged = directory.Combine("staged");
        Directory.CreateDirectory(staged);
        await File.WriteAllBytesAsync(Path.Combine(staged, "model.q8.1.0.0.slm"), [1, 2, 3], TestContext.Current.CancellationToken);
        var file = await PackageFileHasher.DescribeAsync(Path.Combine(staged, "model.q8.1.0.0.slm"), "model.q8.1.0.0.slm", cancellationToken: TestContext.Current.CancellationToken);
        var manifest = new PackageManifest { Id = "model.q8", Version = "1.0.0", Files = [file], Provenance = new PackageProvenance { Sources = pins } };
        await File.WriteAllBytesAsync(Path.Combine(staged, FeedLayout.ManifestFileName), FeedJson.Write(manifest, FeedJsonContext.Default.PackageManifest), TestContext.Current.CancellationToken);
        await new FeedPublisher().PublishAsync(feed, [staged], key, cancellationToken: TestContext.Current.CancellationToken);

        Assert.Null(VersionPlanner.Next(feed, "model.q8", pins, file.Sha256));
        Assert.Equal("1.1.0", VersionPlanner.Next(feed, "model.q8", pins, new string('b', 64)));
        ProvenanceSource[] newPins = [pins[0] with { Revision = new string('2', 40) }];
        Assert.Equal("2.0.0", VersionPlanner.Next(feed, "model.q8", newPins, new string('b', 64)));
    }
}
