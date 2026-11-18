using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.ModelFeed.Index;
using Sharplight.ModelFeed.Resolution;
using Sharplight.ModelFeed.Versioning;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A package of the model feed in Discover: what it is, whether it fits and installs here, and whether it is installed.</summary>
internal sealed partial class FeedPackageViewModel : ObservableObject
{
    public FeedPackageViewModel(FeedPackageSummary summary, CompatibilityEnvironment environment, bool verified)
    {
        Summary = summary;
        IsVerified = verified;
        KindInfo = ModelKindCatalog.For(summary.Model?.Kind ?? "");
        IncompatibleReasons = PackageCompatibility.Check(summary.Requirements, environment);
        SearchText = string.Join(' ', [Name, Id, summary.Display?.Family ?? "", summary.Display?.Variant ?? "", summary.Model?.Architecture ?? "", summary.Model?.Kind ?? "",
            .. summary.Display?.Tags ?? [], .. summary.Model?.Tasks ?? []]);
    }

    /// <summary>The listing entry.</summary>
    public FeedPackageSummary Summary { get; }

    /// <summary>The package id (<c>qwen3-0.6b.q8-0-gguf</c>).</summary>
    public string Id => Summary.Id;

    /// <summary>The display name, else the id.</summary>
    public string Name => Summary.Display?.Name ?? Summary.Id;

    /// <summary>Variant label, e.g. <c>Q8_0 GGUF</c>, or empty.</summary>
    public string Variant => Summary.Display?.Variant ?? "";

    /// <summary>The name and variant in one line.</summary>
    public string Title => Variant.Length == 0 ? Name : $"{Name} · {Variant}";

    /// <summary>One-sentence description, or empty.</summary>
    public string Description => Summary.Display?.Summary ?? "";

    /// <summary>The latest version listed in the channel.</summary>
    public string Version => Summary.LatestVersion;

    /// <summary>The download size.</summary>
    public string Size => ByteFormat.Format(Summary.Size);

    /// <summary>How the kind is presented.</summary>
    public ModelKindInfo KindInfo { get; }

    /// <summary>The kind's name.</summary>
    public string KindTitle => KindInfo.Title;

    /// <summary>The kind's icon resource key.</summary>
    public string KindIcon => KindInfo.Icon;

    /// <summary>The kind id (for the task filter).</summary>
    public string Kind => Summary.Model?.Kind ?? "";

    /// <summary>The index and key set were verified against the trusted key (every package of a feed listing is signed).</summary>
    public bool IsVerified { get; }

    /// <summary>The license name, or empty.</summary>
    public string License => Summary.License?.Name ?? Summary.License?.Spdx ?? "";

    /// <summary>The model's publisher, or empty.</summary>
    public string Publisher => Summary.Publisher?.Name ?? "";

    /// <summary>Architecture, quantization and parameter count, e.g. <c>qwen3 · Q8_0 · 596M parameters</c>.</summary>
    public string Details => string.Join(" · ", new[]
    {
        Summary.Model?.Architecture,
        Summary.Model?.Quantization,
        Summary.Model?.Parameters is { } parameters ? $"{ByteFormat.Count(parameters)} parameters" : null,
    }.Where(part => !string.IsNullOrEmpty(part)));

    /// <summary>The packages this one installs with (required dependencies), one per line.</summary>
    public IReadOnlyList<string> Dependencies => [.. Summary.Dependencies.Where(dependency => dependency.IsRequired).Select(dependency => dependency.Id)];

    /// <summary>The package has required dependencies.</summary>
    public bool HasDependencies => Dependencies.Count > 0;

    /// <summary>Capability and tag badges.</summary>
    public IReadOnlyList<string> Tags => Summary.Display?.Tags ?? [];

    /// <summary>The text the filter searches.</summary>
    public string SearchText { get; }

    /// <summary>Why this host cannot install it (version or format requirements); empty when it can.</summary>
    public IReadOnlyList<string> IncompatibleReasons { get; }

    /// <summary>The package can be installed here.</summary>
    public bool IsInstallable => IncompatibleReasons.Count == 0 && !IsYanked;

    /// <summary>The listing says why it cannot be installed.</summary>
    public bool IsBlocked => !IsInstallable;

    /// <summary>The reason it cannot be installed.</summary>
    public string BlockedText => IsYanked ? $"Withdrawn by the publisher: {Summary.State?.Yanked?.Reason}".TrimEnd(' ', ':') : "This package " + string.Join("; ", IncompatibleReasons);

    /// <summary>The publisher withdrew the latest version.</summary>
    public bool IsYanked => Summary.State?.Yanked is not null;

    /// <summary>The package is deprecated (still installable).</summary>
    public bool IsDeprecated => Summary.State?.Deprecated is not null;

    /// <summary>The deprecation notice, or empty.</summary>
    public string DeprecationText => Summary.State?.Deprecated?.Reason ?? "Deprecated by the publisher.";

    /// <summary>The version in the model store, or null when it is not installed.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsInstalled), nameof(HasUpdate), nameof(InstallText), nameof(InstalledText))]
    public partial string? InstalledVersion { get; set; }

    /// <summary>The package is installed (at any version).</summary>
    public bool IsInstalled => InstalledVersion is not null;

    /// <summary>A newer version than the installed one is listed.</summary>
    public bool HasUpdate => InstalledVersion is { } installed && SemanticVersion.TryParse(installed, out var have) && SemanticVersion.TryParse(Version, out var latest) && latest > have;

    /// <summary>The label of the badge, e.g. <c>Installed 1.0.0</c>.</summary>
    public string InstalledText => InstalledVersion is null ? "" : $"Installed {InstalledVersion}";

    /// <summary>The label of the install button.</summary>
    public string InstallText => HasUpdate ? $"Update to {Version}" : IsInstalled ? "Installed · install again" : "Download and install";

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

    /// <summary>The device memory the package needs (the feed's figure for one slot, else the file size), 0 when unknown.</summary>
    public long MemoryBytes
    {
        get
        {
            var memory = Summary.Requirements?.Memory ?? [];
            var gpu = memory.FirstOrDefault(entry => string.Equals(entry.Device, "gpu", StringComparison.OrdinalIgnoreCase)) ?? memory.FirstOrDefault();
            return gpu is not null ? gpu.TotalFor(1) : Summary.Size;
        }
    }

    /// <summary>Rates the package against the machine's memory (all GPUs together, system memory).</summary>
    public void EstimateFit(long gpuBytes, long systemBytes)
    {
        long need = MemoryBytes;
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
