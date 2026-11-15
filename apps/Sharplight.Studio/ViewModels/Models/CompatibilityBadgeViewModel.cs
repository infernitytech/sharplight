using Sharplight.Studio.Services.Compatibility;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The likely compatibility of a model that is not downloaded yet, as a badge and as the box above the download button.</summary>
internal sealed class CompatibilityBadgeViewModel
{
    /// <param name="compatibility">The verdict.</param>
    public CompatibilityBadgeViewModel(ModelCompatibility compatibility)
    {
        Compatibility = compatibility;
    }

    /// <summary>The verdict.</summary>
    public ModelCompatibility Compatibility { get; }

    /// <summary>Badge text: <c>Supported</c>, <c>Partially supported</c>, <c>Unsupported architecture hrm_text</c>, ...</summary>
    public string Text => Compatibility.Text;

    /// <summary>
    /// A shorter badge for the quantization rows, where the architecture is the repository's: <c>Unsupported Q8_K</c>, <c>Partial</c>,
    /// <c>Architecture supported</c> until the weight formats are read, ...
    /// </summary>
    public string ShortText => Compatibility.State switch
    {
        CompatibilityState.Supported when !Compatibility.WeightsChecked => "Architecture supported",
        CompatibilityState.Partial => "Partial",
        CompatibilityState.UnsupportedArchitecture => "Unsupported architecture",
        CompatibilityState.UnsupportedWeightFormat => $"Unsupported {Compatibility.Subject}",
        CompatibilityState.UnsupportedVariant => "Unsupported variant",
        CompatibilityState.UnsupportedKind => "Not yet supported",
        CompatibilityState.CheckedAtImport => "ONNX",
        _ => Compatibility.Text,
    };

    /// <summary>Expected to import and run.</summary>
    public bool IsGood => Compatibility.State == CompatibilityState.Supported;

    /// <summary>Runs with a restriction, or is a projector rather than a model.</summary>
    public bool IsWarning => Compatibility.State is CompatibilityState.Partial or CompatibilityState.Projector or CompatibilityState.DraftModel;

    /// <summary>Importing or loading is expected to fail.</summary>
    public bool IsBad => Compatibility.IsUnsupported;

    /// <summary>Neither verdict applies yet (unknown metadata, ONNX operators checked at import).</summary>
    public bool IsNeutral => !IsGood && !IsWarning && !IsBad;

    /// <summary>The heading of the box above the download button.</summary>
    public string Title => Compatibility.State switch
    {
        CompatibilityState.Supported => "Supported by Sharplight",
        CompatibilityState.Unknown => "Compatibility unknown",
        CompatibilityState.CheckedAtImport => "ONNX: operators are checked at import",
        _ => Compatibility.Text,
    };

    /// <summary>The reason, and that downloading stays possible when it is likely to fail.</summary>
    public string Detail
    {
        get
        {
            string detail = Compatibility.Reason;
            if (!Compatibility.WeightsChecked && Compatibility.State is CompatibilityState.Supported or CompatibilityState.Partial)
            {
                detail += " The weight formats are read when this quantization is selected.";
            }

            return IsBad && Compatibility.State != CompatibilityState.UnsupportedKind ? detail + " Downloading is still possible." : detail;
        }
    }

    /// <summary>Tooltip of the badge.</summary>
    public string ToolTip => Detail;
}
