using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>One line of a model's size breakdown: what it is, its size and its share of the whole.</summary>
internal sealed class BreakdownLineViewModel
{
    /// <param name="title">What the bytes are.</param>
    /// <param name="detail">Extra information (kernel count), or empty.</param>
    /// <param name="bytes">The size.</param>
    /// <param name="total">The size of the whole model, for the share.</param>
    public BreakdownLineViewModel(string title, string detail, long bytes, long total)
    {
        Title = title;
        Detail = detail;
        Bytes = bytes;
        Fraction = total <= 0 ? 0 : Math.Clamp((double)bytes / total, 0, 1);
    }

    /// <summary>What the bytes are.</summary>
    public string Title { get; }

    /// <summary>Extra information, or empty.</summary>
    public string Detail { get; }

    /// <summary>The size in bytes.</summary>
    public long Bytes { get; }

    /// <summary>The size, e.g. <c>4.1 GB</c>.</summary>
    public string Size => ByteFormat.Format(Bytes);

    /// <summary>The share of the whole model, 0..1.</summary>
    public double Fraction { get; }
}
