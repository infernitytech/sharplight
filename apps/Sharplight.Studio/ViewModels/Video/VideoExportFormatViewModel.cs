namespace Sharplight.Studio.ViewModels.Video;

/// <summary>One file format of the Video page's export buttons: available when the encoder catalog has an encoder for it.</summary>
internal sealed class VideoExportFormatViewModel
{
    /// <param name="format">The format id (<c>Sharplight.Video.VideoFormats</c>).</param>
    /// <param name="label">The button's text.</param>
    /// <param name="extension">The extension of the files, with the dot.</param>
    /// <param name="isAvailable">An encoder for the format is available on this machine.</param>
    /// <param name="tip">The tooltip: what the format is, or why it is unavailable.</param>
    public VideoExportFormatViewModel(string format, string label, string extension, bool isAvailable, string tip)
    {
        Format = format;
        Label = label;
        Extension = extension;
        IsAvailable = isAvailable;
        Tip = tip;
    }

    /// <summary>The format id.</summary>
    public string Format { get; }

    /// <summary>The button's text.</summary>
    public string Label { get; }

    /// <summary>The extension of the files, with the dot.</summary>
    public string Extension { get; }

    /// <summary>An encoder for the format is available on this machine.</summary>
    public bool IsAvailable { get; }

    /// <summary>The tooltip: what the format is, or why it is unavailable.</summary>
    public string Tip { get; }
}
