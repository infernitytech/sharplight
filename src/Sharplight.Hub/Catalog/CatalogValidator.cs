using System.Globalization;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Hub.Catalog;

/// <summary>Checks a <see cref="VerifiedCatalog"/> for consistency without the network (docs/verified-catalog.md, "Rules").</summary>
public static partial class CatalogValidator
{
    /// <summary>The task kinds an entry may have.</summary>
    public static IReadOnlyList<string> Kinds { get; } =
    [
        ModelKinds.TextGeneration, ModelKinds.SpeechRecognition, ModelKinds.SpeechSynthesis, ModelKinds.ImageGeneration, ModelKinds.VideoGeneration, ModelKinds.VideoEnhancement,
        ModelKinds.ImageClassification, ModelKinds.ImageEmbedding, ModelKinds.ObjectDetection, ModelKinds.ImageSegmentation,
        ModelKinds.SemanticSegmentation, ModelKinds.DepthEstimation, ModelKinds.Ocr, ModelKinds.ImageUpscaling, ModelKinds.AudioCodec,
        ModelKinds.VoiceActivity, ModelKinds.AudioSeparation, ModelKinds.AudioEmbedding, ModelKinds.MusicGeneration, ModelKinds.AudioGeneration, EmbeddingsKind, RerankerKind,
    ];

    /// <summary>The companion roles.</summary>
    public static IReadOnlyList<string> Roles { get; } =
    [
        AuxiliaryRoles.Mmproj, AuxiliaryRoles.Draft, AuxiliaryRoles.Vocoder, AuxiliaryRoles.TextEncoder, AuxiliaryRoles.SecondTextEncoder,
        AuxiliaryRoles.ThirdTextEncoder, AuxiliaryRoles.Vae, AuxiliaryRoles.Lora, AuxiliaryRoles.SecondTransformer, AuxiliaryRoles.ImageEncoder,
        AuxiliaryRoles.TinyVae, AuxiliaryRoles.ControlNet,
    ];

    /// <summary>The kind of text embedding models (not a <see cref="ModelKinds"/> value).</summary>
    public const string EmbeddingsKind = "embeddings";

    /// <summary>The kind of reranking models, cross-encoders and decoder rerankers (not a <see cref="ModelKinds"/> value).</summary>
    public const string RerankerKind = "reranker";

    /// <summary>The longest <see cref="CatalogEntry.LicenseNotice"/>: it is shown as a callout, not a licence text.</summary>
    public const int MaxLicenseNoticeLength = 600;

    /// <summary>Every problem found; empty when the catalog is consistent.</summary>
    /// <param name="catalog">The catalog to check.</param>
    public static IReadOnlyList<string> Validate(VerifiedCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        var problems = new List<string>();
        if (catalog.SchemaVersion != VerifiedCatalog.CurrentSchemaVersion)
        {
            problems.Add($"schemaVersion {catalog.SchemaVersion} is not {VerifiedCatalog.CurrentSchemaVersion}");
        }

        if (!IsDate(catalog.Updated))
        {
            problems.Add($"updated '{catalog.Updated}' is not a yyyy-MM-dd date");
        }

        var sections = new HashSet<string>(StringComparer.Ordinal);
        foreach (var section in catalog.DocSections)
        {
            if (!sections.Add(section.Id))
            {
                problems.Add($"doc section {section.Id} is defined twice");
            }

            if (section.Columns.Count == 0 || section.Columns.Select(column => column.Key).Distinct().Count() != section.Columns.Count)
            {
                problems.Add($"doc section {section.Id} needs distinct columns");
            }
        }

        var ids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var entry in catalog.Entries)
        {
            if (!ids.Add(entry.Id))
            {
                problems.Add($"entry id {entry.Id} is used twice");
            }

            ValidateEntry(entry, sections, problems);
        }

        return problems;
    }

    private static void ValidateEntry(CatalogEntry entry, HashSet<string> sections, List<string> problems)
    {
        void Problem(string text) => problems.Add($"{entry.Id}: {text}");
        if (!IdPattern().IsMatch(entry.Id))
        {
            Problem("the id must be lower-case letters, digits, '.' and '-'");
        }

        if (string.IsNullOrWhiteSpace(entry.Name))
        {
            Problem("the name is empty");
        }

        if (entry.LicenseNotice is { } licenseNotice && (string.IsNullOrWhiteSpace(licenseNotice) || licenseNotice.Length > MaxLicenseNoticeLength))
        {
            Problem($"licenseNotice must be one to {MaxLicenseNoticeLength} characters");
        }

        if (CatalogLicenses.NeedsNotice(entry.License) && string.IsNullOrWhiteSpace(entry.LicenseNotice))
        {
            Problem($"the license '{entry.License}' is not permissive: add a licenseNotice");
        }

        if (!Kinds.Contains(entry.Kind))
        {
            Problem($"unknown kind '{entry.Kind}'");
        }

        if (entry.Verification.Date.Length > 0 && !IsDate(entry.Verification.Date))
        {
            Problem($"verification date '{entry.Verification.Date}' is not yyyy-MM-dd");
        }

        if (entry.IsInstallable && (entry.Verification.Date.Length == 0 || entry.Verification.Summary.Length == 0 || entry.Verification.Providers.Count == 0))
        {
            Problem("an installable entry needs a verification summary, providers and date");
        }

        if (entry.Variants.Count(variant => variant.Recommended) > 1)
        {
            Problem("more than one variant is recommended");
        }

        var roles = entry.Variants.Select(variant => variant.Companions.SelectMany(id => entry.Companions.Where(companion => companion.Id == id)).SelectMany(companion => companion.AllRoles).ToList());
        if (roles.Any(list => list.Count != list.Distinct().Count()))
        {
            Problem("a variant fills a companion role twice");
        }

        foreach (var row in entry.Doc.Where(row => !sections.Contains(row.Section)))
        {
            Problem($"doc row names the unknown section {row.Section}");
        }

        var variantIds = new HashSet<string>(StringComparer.Ordinal);
        var variantLabels = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var companionIds = entry.Companions.Select(companion => companion.Id).ToHashSet(StringComparer.Ordinal);
        foreach (var variant in entry.Variants)
        {
            if (!variantIds.Add(variant.Id) || !IdPattern().IsMatch(variant.Id))
            {
                Problem($"variant id '{variant.Id}' is invalid or used twice");
            }

            if (!CatalogFormats.All.Contains(variant.Format))
            {
                Problem($"variant {variant.Id}: unknown format '{variant.Format}'");
            }

            if (string.IsNullOrWhiteSpace(variant.Label))
            {
                Problem($"variant {variant.Id}: the label is empty");
            }
            else if (!variantLabels.Add(variant.Label))
            {
                // Studio lists the variants by label: two with one label cannot be told apart.
                Problem($"variant {variant.Id}: the label '{variant.Label}' is used twice");
            }

            foreach (var id in variant.Companions.Where(id => !companionIds.Contains(id)))
            {
                Problem($"variant {variant.Id}: unknown companion '{id}'");
            }

            ValidateFiles($"variant {variant.Id}", variant, Problem);
            if (variant.Level is { } level && (level is not (CatalogLevel.Verified or CatalogLevel.Partial) || level < entry.Level))
            {
                Problem($"variant {variant.Id}: level {level} must be verified or partial and not above the entry's level {entry.Level}");
            }

            if (variant.Level == CatalogLevel.Partial && entry.Level != CatalogLevel.Partial && string.IsNullOrWhiteSpace(variant.Verification))
            {
                Problem($"variant {variant.Id}: a partial variant of a verified entry needs a verification note (what was and was not checked)");
            }

            if (variant.Quality is { } quality && !CatalogQualityTiers.All.Contains(quality, StringComparer.Ordinal))
            {
                Problem($"variant {variant.Id}: unknown quality '{quality}' ({string.Join(", ", CatalogQualityTiers.All)})");
            }
        }

        if (companionIds.Count != entry.Companions.Count)
        {
            Problem("companion ids must be distinct");
        }

        foreach (var companion in entry.Companions)
        {
            foreach (var role in companion.AllRoles.Where(role => !Roles.Contains(role)))
            {
                Problem($"companion {companion.Id}: unknown role '{role}'");
            }

            if (!entry.Variants.Any(variant => variant.Companions.Contains(companion.Id)))
            {
                Problem($"companion {companion.Id} is not used by any variant");
            }

            ValidateFiles($"companion {companion.Id}", companion, Problem);
            foreach (var variant in entry.Variants.Where(variant => companion.IntoVariantFolder && variant.Companions.Contains(companion.Id)))
            {
                foreach (var file in companion.Files.Where(file => variant.Files.Any(own => string.Equals(own.Path, file.Path, StringComparison.OrdinalIgnoreCase))))
                {
                    Problem($"companion {companion.Id}: '{file.Path}' goes into the folder of variant {variant.Id}, which has a file of that name");
                }
            }
        }
    }

    private static void ValidateFiles(string what, CatalogFileSet set, Action<string> problem)
    {
        if (!RepositoryPattern().IsMatch(set.Repository))
        {
            problem($"{what}: '{set.Repository}' is not a repository id");
        }

        if (!CommitPattern().IsMatch(set.Revision))
        {
            problem($"{what}: revision '{set.Revision}' is not a 40-digit lower-case commit");
        }

        if (set.Source is { } source && !CatalogSources.All.Contains(source, StringComparer.Ordinal))
        {
            problem($"{what}: unknown source '{source}' ({string.Join(", ", CatalogSources.All)})");
        }

        if (set.Source == CatalogSources.GitHubRelease && string.IsNullOrWhiteSpace(set.Release))
        {
            problem($"{what}: a github-release set needs the release tag");
        }

        if (set.Release is not null && !CatalogSources.IsGitHub(set.Source))
        {
            problem($"{what}: a release tag needs a GitHub source");
        }

        if (set.Files.Count == 0)
        {
            problem($"{what}: no files");
            return;
        }

        var paths = new HashSet<string>(StringComparer.Ordinal);
        foreach (var file in set.Files)
        {
            if (!paths.Add(file.Path) || file.Path.Length == 0 || file.Path.StartsWith('/') || file.Path.Contains("..", StringComparison.Ordinal))
            {
                problem($"{what}: file path '{file.Path}' is invalid or listed twice");
            }

            if (file.Size <= 0)
            {
                problem($"{what}: {file.Path} has no size");
            }

            bool hasSha = file.Sha256 is { } sha && Sha256Pattern().IsMatch(sha);
            bool hasOid = file.GitOid is { } oid && CommitPattern().IsMatch(oid);
            if (!hasSha && !hasOid)
            {
                problem($"{what}: {file.Path} needs a sha256 (LFS) or a git oid");
            }
            else if (!hasSha && !set.IsHubSource)
            {
                // GitHub lists no content checksums (release assets aside): the catalog records the sha256 the files were verified with.
                problem($"{what}: {file.Path} needs a sha256 (files from GitHub)");
            }
        }

        bool primaryKnown = set.Primary.Length == 0 || paths.Contains(set.Primary) || paths.Any(path => path.StartsWith(set.Primary + "/", StringComparison.Ordinal));
        if (!primaryKnown)
        {
            problem($"{what}: primary '{set.Primary}' is neither a listed file nor a directory of one");
        }
    }

    private static bool IsDate(string text) => DateOnly.TryParseExact(text, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out _);

    [GeneratedRegex(@"^[a-z0-9][a-z0-9.-]*$")]
    private static partial Regex IdPattern();

    [GeneratedRegex(@"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")]
    private static partial Regex RepositoryPattern();

    [GeneratedRegex(@"^[0-9a-f]{40}$")]
    private static partial Regex CommitPattern();

    [GeneratedRegex(@"^[0-9a-f]{64}$")]
    private static partial Regex Sha256Pattern();
}
