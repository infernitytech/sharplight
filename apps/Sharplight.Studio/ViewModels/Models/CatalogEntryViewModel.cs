using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Hub.Catalog;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A model of the verified catalog in Discover: what it is, how it was verified, its variants.</summary>
internal sealed partial class CatalogEntryViewModel : ObservableObject
{
    public CatalogEntryViewModel(CatalogEntry entry)
    {
        Entry = entry;
        Variants = [.. entry.Variants.Select(variant => new CatalogVariantViewModel(CatalogInstallPlan.For(entry, variant)))];
        long smallest = Variants.Min(variant => variant.Plan.TotalBytes);
        long largest = Variants.Max(variant => variant.Plan.TotalBytes);
        SizeRange = smallest == largest ? ByteFormat.Format(smallest) : $"{ByteFormat.Format(smallest)} – {ByteFormat.Format(largest)}";
        SearchText = string.Join(' ', [entry.Name, entry.Family, entry.Architecture ?? "", .. entry.Tags, .. entry.Variants.SelectMany(variant => new[] { variant.Repository, variant.Label })]);
    }

    public CatalogEntry Entry { get; }

    public IReadOnlyList<CatalogVariantViewModel> Variants { get; }

    public string Name => Entry.Name;

    public string Family => Entry.Family;

    public string KindTitle => ModelKindCatalog.For(Entry.Kind).Title;

    public string KindIcon => ModelKindCatalog.For(Entry.Kind).Icon;

    /// <summary>The verification result.</summary>
    public string Verification => Entry.Verification.Summary;

    /// <summary><c>Verified 2026-11-16 against llama.cpp · CPU, Vulkan</c>.</summary>
    public string VerifiedText
    {
        get
        {
            var verification = Entry.Verification;
            var reference = verification.Reference is { Length: > 0 } name ? $" against {name}" : "";
            var providers = verification.Providers.Count > 0 ? $" · {string.Join(", ", verification.Providers)}" : "";
            return $"{(Entry.Level == CatalogLevel.Partial ? "Partially verified" : "Verified")} {verification.Date}{reference}{providers}";
        }
    }

    public string? License => Entry.License;

    public bool HasLicense => !string.IsNullOrEmpty(Entry.License);

    public bool Gated => Entry.Gated;

    public bool IsPartial => Entry.Level == CatalogLevel.Partial;

    public string? Notes => Entry.Notes;

    /// <summary>The licence needs a prominent notice (<c>other</c>, non-commercial): the licence and the notes.</summary>
    public string? LicenseNotice => LicenseNotices.For(Entry);

    /// <summary>The entry has a licence notice.</summary>
    public bool HasLicenseNotice => LicenseNotice is not null;

    /// <summary>The notes are shown as the plain callout (the licence notice already carries them).</summary>
    public bool HasPlainNotes => HasNotes && !HasLicenseNotice;

    public bool HasNotes => !string.IsNullOrEmpty(Entry.Notes);

    public IReadOnlyList<string> Tags => Entry.Tags;

    /// <summary>The download sizes of the variants (companions included), smallest to largest.</summary>
    public string SizeRange { get; }

    /// <summary><c>3 variants</c>.</summary>
    public string VariantCount => ByteFormat.Count(Variants.Count, "variant", "variants");

    /// <summary>The text a search matches.</summary>
    public string SearchText { get; }

    /// <summary>A variant of the entry is installed.</summary>
    [ObservableProperty]
    public partial bool IsInstalled { get; set; }
}
