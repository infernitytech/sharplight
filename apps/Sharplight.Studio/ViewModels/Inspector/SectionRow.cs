using System.Globalization;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>A region of the package file: the header, a section (of any kind, known or not), the section table, or unused bytes.</summary>
/// <param name="Kind">Four-character kind (<c>DEFN</c>, <c>ARTS</c>, ...), a pseudo kind (<see cref="SectionViews.HeaderKind"/>, ...), or the number of an unknown kind.</param>
/// <param name="Name">Section name (artifact-set id, definition layout), empty when none.</param>
/// <param name="Offset">Absolute offset.</param>
/// <param name="Length">Length in bytes.</param>
/// <param name="Percent">Share of the file, 0 to 100.</param>
internal sealed record SectionRow(string Kind, string Name, long Offset, long Length, double Percent)
{
    /// <summary>How the inspector presents the kind.</summary>
    public SectionView View => SectionViews.For(Kind);

    /// <summary>Readable kind.</summary>
    public string Title => View.Title;

    /// <summary>What the region holds.</summary>
    public string Description => View.Description;

    /// <summary>False for sections of a newer format (shown raw).</summary>
    public bool IsDecoded => View.IsDecoded;

    /// <summary>True when another tab shows the decoded contents.</summary>
    public bool CanShow => View.Tab != InspectorTab.Sections;

    /// <summary>Label of the button that opens that tab.</summary>
    public string ShowLabel => $"Show in {View.Tab}";

    /// <summary>The offset for display.</summary>
    public string OffsetText => ByteFormat.Number(Offset);

    /// <summary>The end offset (exclusive) for display.</summary>
    public string EndText => ByteFormat.Number(Offset + Length);

    /// <summary>The size for display.</summary>
    public string Size => ByteFormat.Format(Length);

    /// <summary>The exact size for display.</summary>
    public string ExactSize => ByteFormat.Exact(Length);

    /// <summary>The share for display.</summary>
    public string Share => Percent >= 0.1 || Percent == 0 ? Percent.ToString("0.0", CultureInfo.InvariantCulture) + " %" : "< 0.1 %";
}
