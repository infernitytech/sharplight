using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Hub.Catalog;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A variant of a verified-catalog entry: what installing it downloads, how much memory it needs and whether it is installed.</summary>
internal sealed partial class CatalogVariantViewModel : ObservableObject
{
    public CatalogVariantViewModel(CatalogInstallPlan plan)
    {
        Plan = plan;
        Companions = [.. plan.Companions.Select(companion => $"{companion.Label} · {ByteFormat.Format(companion.TotalBytes)}")];
    }

    /// <summary>What installing the variant does.</summary>
    public CatalogInstallPlan Plan { get; }

    /// <summary>The variant.</summary>
    public CatalogVariant Variant => Plan.Variant;

    public string Label => Variant.Label;

    public bool Recommended => Variant.Recommended;

    /// <summary>The download size, companions included.</summary>
    public string Size => ByteFormat.Format(Plan.TotalBytes);

    /// <summary>Files and repository, e.g. <c>3 files · city96/FLUX.1-schnell-gguf</c>.</summary>
    public string Details => $"{ByteFormat.Count(Plan.FileCount, "file", "files")} · {Variant.Repository}";

    /// <summary>The companions installed with it (label and size), in order.</summary>
    public IReadOnlyList<string> Companions { get; }

    public bool HasCompanions => Companions.Count > 0;

    public string? Notes => Variant.Notes;

    public bool HasNotes => !string.IsNullOrEmpty(Variant.Notes);

    /// <summary>The approximate device memory needed.</summary>
    public string Memory => Variant.MemoryBytes > 0 ? $"Needs about {ByteFormat.Format(Variant.MemoryBytes)} of device memory (estimate, companions included)" : "";

    [ObservableProperty]
    public partial bool IsInstalled { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasFit), nameof(IsGood), nameof(IsWarning), nameof(IsBad))]
    public partial string FitText { get; set; } = "";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsGood), nameof(IsWarning), nameof(IsBad))]
    public partial CatalogFit Fit { get; set; } = CatalogFit.Unknown;

    public bool HasFit => FitText.Length > 0;

    public bool IsGood => Fit == CatalogFit.Gpu;

    public bool IsWarning => Fit == CatalogFit.Cpu;

    public bool IsBad => Fit == CatalogFit.TooLarge;

    /// <summary>Rates the variant against the machine's memory: GPU memory (all GPUs, which a model can be split over) and system memory.</summary>
    /// <param name="gpuBytes">Total memory of the GPUs, 0 without one.</param>
    /// <param name="systemBytes">System memory, 0 when unknown.</param>
    public void EstimateFit(long gpuBytes, long systemBytes)
    {
        long need = Variant.MemoryBytes;
        if (need <= 0 || (gpuBytes <= 0 && systemBytes <= 0))
        {
            Fit = CatalogFit.Unknown;
            FitText = "";
            return;
        }

        (Fit, FitText) = need <= gpuBytes ? (CatalogFit.Gpu, "Fits your GPU")
            : need <= systemBytes ? (CatalogFit.Cpu, gpuBytes > 0 ? "CPU only" : "Runs on the CPU")
            : (CatalogFit.TooLarge, "Too large");
    }
}
