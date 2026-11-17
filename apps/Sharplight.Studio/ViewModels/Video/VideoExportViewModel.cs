using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Video;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>
/// The Video page's export settings: the file format, then the codec, preset, pixel format and quality that format's encoder offers
/// (<see cref="IVideoEncoder.Codecs"/>, read off the UI thread because the first read tries the hardware encoders).
/// </summary>
internal sealed partial class VideoExportViewModel : ObservableObject
{
    private readonly VideoEncoderCatalog _encoders;

    /// <param name="encoders">The encoders of the formats.</param>
    public VideoExportViewModel(VideoEncoderCatalog encoders)
    {
        _encoders = encoders;
        Formats = CreateFormats(encoders);
        SelectedFormat = Formats.FirstOrDefault(format => format.IsAvailable);
    }

    /// <summary>The file formats, available or not (the tooltip says why not).</summary>
    public IReadOnlyList<VideoExportFormatViewModel> Formats { get; }

    /// <summary>The format to export.</summary>
    [ObservableProperty]
    public partial VideoExportFormatViewModel? SelectedFormat { get; set; }

    /// <summary>The codecs of the selected format's encoder (empty for formats without a choice).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasCodecs))]
    public partial IReadOnlyList<VideoCodecInfo> Codecs { get; private set; } = [];

    /// <summary>The format offers a codec choice.</summary>
    public bool HasCodecs => Codecs.Count > 0;

    /// <summary>The codec to encode with.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Presets), nameof(HasPresets), nameof(PixelFormats), nameof(HasPixelFormats), nameof(ShowQuality), nameof(CodecDescription))]
    public partial VideoCodecInfo? SelectedCodec { get; set; }

    /// <summary>What the selected codec is good for.</summary>
    public string CodecDescription => SelectedCodec?.Description ?? "";

    /// <summary>The presets of the selected codec.</summary>
    public IReadOnlyList<string> Presets => SelectedCodec?.Presets ?? [];

    /// <summary>The codec has presets.</summary>
    public bool HasPresets => Presets.Count > 0;

    /// <summary>The preset to use.</summary>
    [ObservableProperty]
    public partial string? SelectedPreset { get; set; }

    /// <summary>The pixel formats of the selected codec.</summary>
    public IReadOnlyList<string> PixelFormats => SelectedCodec?.PixelFormats ?? [];

    /// <summary>The codec offers more than one pixel format.</summary>
    public bool HasPixelFormats => PixelFormats.Count > 1;

    /// <summary>The pixel format to encode.</summary>
    [ObservableProperty]
    public partial string? SelectedPixelFormat { get; set; }

    /// <summary>Visual quality 0-100 (ignored by lossless codecs and when a bitrate is set).</summary>
    [ObservableProperty]
    public partial double Quality { get; set; } = 80;

    /// <summary>Target bitrate in kbit/s; 0 encodes at constant quality.</summary>
    [ObservableProperty]
    public partial double BitrateKilobitsPerSecond { get; set; }

    /// <summary>The quality slider applies (not for lossless codecs).</summary>
    public bool ShowQuality => SelectedCodec is not { IsLossless: true };

    /// <summary>The codec list of the current format has been read (for tests).</summary>
    internal Task CodecsLoaded { get; private set; } = Task.CompletedTask;

    /// <summary>The encode options of the current choices.</summary>
    /// <returns>The options.</returns>
    public VideoEncodeOptions ToOptions() => new()
    {
        Codec = SelectedCodec?.Id,
        Preset = HasPresets ? SelectedPreset : null,
        PixelFormat = HasPixelFormats ? SelectedPixelFormat : null,
        Quality = (int)Math.Round(Quality),
        BitrateKilobitsPerSecond = BitrateKilobitsPerSecond > 0 ? (int)BitrateKilobitsPerSecond : null,
    };

    partial void OnSelectedFormatChanged(VideoExportFormatViewModel? value)
    {
        Codecs = [];
        SelectedCodec = null;
        if (value is null || !_encoders.TryFind(value.Format, out IVideoEncoder? encoder))
        {
            return;
        }

        CodecsLoaded = LoadCodecsAsync(value, encoder);
    }

    partial void OnSelectedCodecChanged(VideoCodecInfo? value)
    {
        SelectedPreset = value?.DefaultPreset ?? value?.Presets.FirstOrDefault();
        SelectedPixelFormat = value?.PixelFormats.FirstOrDefault();
    }

    private async Task LoadCodecsAsync(VideoExportFormatViewModel format, IVideoEncoder encoder)
    {
        IReadOnlyList<VideoCodecInfo> codecs = await Task.Run(() => encoder.Codecs).ConfigureAwait(true);
        if (SelectedFormat != format)
        {
            return;
        }

        Codecs = codecs;
        SelectedCodec = codecs.FirstOrDefault();
    }

    private static VideoExportFormatViewModel[] CreateFormats(VideoEncoderCatalog encoders)
    {
        const string missing = "The bundled FFmpeg libraries are missing from this installation";
        bool Has(string format) => encoders.TryFind(format, out _);
        VideoExportFormatViewModel Bundled(string format, string label, string extension, string tip) =>
            new(format, label, extension, Has(format), Has(format) ? tip : missing);
        return
        [
            Bundled(VideoFormats.Mp4, "MP4", ".mp4", "H.264, AV1 or HEVC (hardware) video: plays everywhere"),
            Bundled(VideoFormats.WebM, "WebM", ".webm", "VP9, AV1 or VP8 video: royalty-free, for the web"),
            Bundled(VideoFormats.Mov, "MOV", ".mov", "ProRes for editing, or H.264"),
            Bundled(VideoFormats.Matroska, "MKV", ".mkv", "Any codec, including lossless FFV1"),
            new(VideoFormats.Gif, "GIF", ".gif", Has(VideoFormats.Gif), "Animated GIF (256 colors)"),
            new(VideoFormats.Apng, "APNG", ".apng", Has(VideoFormats.Apng), "Animated PNG (lossless)"),
        ];
    }
}
