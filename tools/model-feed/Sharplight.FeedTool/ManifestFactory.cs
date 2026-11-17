using Sharplight.Hub.Catalog;
using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Versioning;

namespace Sharplight.FeedTool;

/// <summary>Writes a package manifest from a verified-catalog variant, the built package and the feed overlay.</summary>
internal static class ManifestFactory
{
    /// <summary>The packager name in manifests.</summary>
    public const string PackagerName = "Sharplight";

    private static readonly Dictionary<string, string> TaskOfKind = new(StringComparer.Ordinal)
    {
        ["speech-recognition"] = PackageTasks.Transcription,
        ["speech-synthesis"] = PackageTasks.Speech,
        ["voice-activity"] = PackageTasks.VoiceActivity,
        ["image-generation"] = PackageTasks.ImageGeneration,
        ["video-generation"] = PackageTasks.VideoGeneration,
        ["image-embedding"] = PackageTasks.ImageEmbedding,
        ["image-classification"] = PackageTasks.ImageClassification,
        ["object-detection"] = PackageTasks.ObjectDetection,
        ["image-segmentation"] = PackageTasks.ImageSegmentation,
        ["image-segmentation-semantic"] = PackageTasks.ImageSegmentation,
        ["depth-estimation"] = PackageTasks.DepthEstimation,
        ["ocr"] = PackageTasks.Ocr,
        ["image-upscaling"] = PackageTasks.ImageUpscaling,
        ["audio-codec"] = PackageTasks.AudioCodec,
    };

    private static readonly string[] CapabilityTags = [PackageCapabilities.Tools, PackageCapabilities.Reasoning, PackageCapabilities.Vision];

    /// <summary>The package id of a catalog variant: <c>&lt;entry&gt;.&lt;variant&gt;</c>.</summary>
    /// <param name="entry">Catalog entry.</param>
    /// <param name="variant">Catalog variant.</param>
    /// <returns>The id.</returns>
    public static string PackageId(CatalogEntry entry, CatalogVariant variant) => $"{entry.Id}.{variant.Id}".ToLowerInvariant();

    /// <summary>Builds the manifest.</summary>
    /// <param name="plan">The catalog variant with its companions.</param>
    /// <param name="inspection">What the built package says about itself.</param>
    /// <param name="file">The package file entry (name, size, hashes).</param>
    /// <param name="version">The package version.</param>
    /// <param name="overlay">Feed-only data, or <see langword="null"/>.</param>
    /// <param name="engineVersion">The engine version that built the package.</param>
    /// <param name="feed">The feed the package is published to, or <see langword="null"/>.</param>
    /// <param name="built">Build time.</param>
    /// <returns>The manifest.</returns>
    public static PackageManifest Create(CatalogInstallPlan plan, PackageInspection inspection, PackageFile file, string version, FeedOverlayPackage? overlay, string engineVersion, string? feed, DateTimeOffset built)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(inspection);
        var entry = plan.Entry;
        var variant = plan.Variant;
        var (tasks, capabilities, modalities) = Describe(entry);
        var engine = SemanticVersion.TryParse(engineVersion, out var parsedEngine) ? parsedEngine : SemanticVersion.Create(0, 0, 0);
        var memory = variant.MemoryBytes is > 0
            ? new List<MemoryRequirement>
            {
                new() { Device = "gpu", Weights = inspection.WeightBytes, Overhead = Math.Max(0, variant.MemoryBytes - inspection.WeightBytes) },
                new() { Device = "cpu", Weights = inspection.WeightBytes, Overhead = Math.Max(0, variant.MemoryBytes - inspection.WeightBytes) },
            }
            : [];
        return new PackageManifest
        {
            Id = PackageId(entry, variant),
            Version = version,
            Display = new PackageDisplay { Name = entry.Name, Variant = variant.Label, Family = entry.Family.Length == 0 ? null : entry.Family, Summary = overlay?.Summary, Tags = entry.Tags },
            Publisher = entry.Publisher is { } publisher ? new PackagePublisher { Name = publisher.Name, CountryOfOrigin = publisher.CountryOfOrigin, Url = publisher.Url } : null,
            Packager = new PackagePackager { Name = PackagerName, Feed = feed },
            License = LicenseCatalog.Describe(entry.License, entry.Gated),
            Model = new PackageModel
            {
                Kind = entry.Kind,
                Tasks = tasks,
                Capabilities = capabilities,
                Architecture = entry.Architecture ?? inspection.Architecture,
                Parameters = inspection.Parameters > 0 ? inspection.Parameters : null,
                Quantization = variant.Quantization,
                ContextLength = inspection.ContextLength,
                Languages = overlay?.Languages ?? [],
                Modalities = modalities,
                Defaults = inspection.HasChatTemplate ? new PackageDefaults { ChatTemplate = "embedded" } : null,
            },
            Requirements = new PackageRequirements
            {
                Server = overlay?.Server,
                Engine = $">={engine.Major}.{engine.Minor}.0",
                PackageFormat = inspection.FormatVersion,
                Memory = memory,
                Disk = file.Size,
            },
            Dependencies = overlay?.Dependencies ?? [],
            Provenance = new PackageProvenance
            {
                CatalogEntry = entry.Id,
                CatalogVariant = variant.Id,
                Sources = [Source(variant.Repository, variant.Revision, variant.Files, "primary"), .. plan.Companions.Select(companion => Source(companion.Repository, companion.Revision, companion.Files, string.Join(",", companion.AllRoles)))],
                Converter = new ConverterInfo { Engine = engineVersion, Tool = $"feedtool {engineVersion}", Options = new Dictionary<string, string> { ["optimize"] = "true" } },
                Built = built,
            },
            Verification = new PackageVerification
            {
                Level = entry.Level.ToString().ToLowerInvariant(),
                Reference = entry.Verification.Reference,
                Summary = entry.Verification.Summary,
                Providers = entry.Verification.Providers,
                Date = entry.Verification.Date,
                PackageCheck = new PackageCheck { Loaded = true },
            },
            Files = [file],
            State = new PackageState { Listed = overlay?.Listed ?? true },
            Notes = entry.Notes,
        };
    }

    /// <summary>Tasks, capabilities and modalities of a catalog entry.</summary>
    /// <param name="entry">Catalog entry.</param>
    /// <returns>The model description parts.</returns>
    public static (IReadOnlyList<string> Tasks, IReadOnlyList<string> Capabilities, PackageModalities Modalities) Describe(CatalogEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        var capabilities = entry.Tags.Where(tag => CapabilityTags.Contains(tag, StringComparer.Ordinal)).ToList();
        switch (entry.Kind)
        {
            case "text-generation":
                capabilities.Add(PackageCapabilities.StructuredOutput);
                var input = capabilities.Contains(PackageCapabilities.Vision) ? new[] { "text", "image" } : ["text"];
                return ([PackageTasks.Chat, PackageTasks.Completion], capabilities, new PackageModalities { Input = input, Output = ["text"] });
            case "embeddings" when entry.Tags.Contains("reranker", StringComparer.Ordinal):
                return ([PackageTasks.Rerank], capabilities, new PackageModalities { Input = ["text"], Output = ["score"] });
            case "embeddings":
                return ([PackageTasks.Embedding], capabilities, new PackageModalities { Input = ["text"], Output = ["embedding"] });
            case "speech-recognition":
                return ([PackageTasks.Transcription], capabilities, new PackageModalities { Input = ["audio"], Output = ["text"] });
            case "speech-synthesis":
                return ([PackageTasks.Speech], capabilities, new PackageModalities { Input = ["text"], Output = ["audio"] });
            case "voice-activity":
                return ([PackageTasks.VoiceActivity], capabilities, new PackageModalities { Input = ["audio"], Output = ["segments"] });
            default:
                return ([TaskOfKind.GetValueOrDefault(entry.Kind, entry.Kind)], capabilities, new PackageModalities());
        }
    }

    private static ProvenanceSource Source(string repository, string revision, IReadOnlyList<CatalogFile> files, string role) => new()
    {
        Repository = repository,
        Revision = revision,
        Role = role,
        Files = [.. files.Select(file => new ProvenanceFile { Path = file.Path, Size = file.Size, Sha256 = file.Sha256, GitOid = file.GitOid })],
    };
}
