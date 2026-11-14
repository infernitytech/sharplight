using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Hub;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A downloadable variant in the repository details.</summary>
internal sealed class VariantViewModel
{
    public VariantViewModel(VariantInfo info, bool stored)
    {
        Info = info;
        IsStored = stored;
    }

    /// <summary>The variant.</summary>
    public VariantInfo Info { get; }

    /// <summary>Tag, or <c>default</c>.</summary>
    public string Tag => Info.Tag.Length == 0 ? "default" : Info.Tag;

    /// <summary>Download size.</summary>
    public string Size => ByteFormat.Format(Info.Bytes);

    /// <summary>Files.</summary>
    public string Files => Info.FileCount == 1 ? "1 file" : $"{Info.FileCount} files";

    /// <summary>Includes a vision projector.</summary>
    public bool HasProjector => Info.HasProjector;

    /// <summary>Pulled when no tag is given.</summary>
    public bool Recommended => Info.Recommended;

    /// <summary>Already in the library.</summary>
    public bool IsStored { get; }
}
