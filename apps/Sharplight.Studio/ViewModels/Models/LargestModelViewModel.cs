using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A row of the storage page's list of the biggest models.</summary>
internal sealed class LargestModelViewModel
{
    public LargestModelViewModel(LibraryEntryViewModel entry, long largest)
    {
        Name = entry.Name;
        Kind = entry.KindTitle;
        Bytes = entry.SizeBytes;
        Fraction = largest <= 0 ? 0 : Math.Clamp((double)entry.SizeBytes / largest, 0, 1);
        LastUsedText = entry.LastUsedText;
        IsPrecompiled = entry.IsPrecompiled;
    }

    /// <summary>The model's name.</summary>
    public string Name { get; }

    /// <summary>The kind's name.</summary>
    public string Kind { get; }

    /// <summary>The size in bytes.</summary>
    public long Bytes { get; }

    /// <summary>The size, e.g. <c>4.1 GB</c>.</summary>
    public string Size => ByteFormat.Format(Bytes);

    /// <summary>The size relative to the biggest model, 0..1.</summary>
    public double Fraction { get; }

    /// <summary>When it was last loaded.</summary>
    public string LastUsedText { get; }

    /// <summary>It has precompiled artifacts inside.</summary>
    public bool IsPrecompiled { get; }
}
