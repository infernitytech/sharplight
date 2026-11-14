using System.Globalization;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>A part of a whole with its share, drawn as a bar (bytes per section kind, weights per type).</summary>
/// <param name="Label">The part.</param>
/// <param name="Detail">Secondary text (counts).</param>
/// <param name="Bytes">Its size.</param>
/// <param name="Percent">Its share of the whole, 0 to 100.</param>
internal sealed record ShareRow(string Label, string Detail, long Bytes, double Percent)
{
    /// <summary>The size for display.</summary>
    public string Size => ByteFormat.Format(Bytes);

    /// <summary>The share for display.</summary>
    public string Share => Percent >= 0.1 || Percent == 0 ? Percent.ToString("0.0", CultureInfo.InvariantCulture) + " %" : "< 0.1 %";
}
