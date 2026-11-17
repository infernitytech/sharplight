using FFmpeg.AutoGen;

namespace Sharplight.Video.Tests;

/// <summary>
/// The bundled FFmpeg libraries must be part of every build output that references Sharplight.Video, so encoding never silently depends on
/// a system installation. These tests fail (not skip) when the libraries are missing: build them with native/ffmpeg/build.sh or download
/// the native-ffmpeg-* CI artifacts into native/ffmpeg/out/.
/// </summary>
public sealed class BundledFfmpegTests
{
    [Fact]
    public void LibrariesAreInTheBuildOutputAndLoad()
    {
        string rid = Ffmpeg.FfmpegLoader.RuntimeIdentifier;
        string directory = Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native");
        string[] missing = [.. BundledFfmpeg.LibraryFileNames.Where(name => !File.Exists(Path.Combine(directory, name)))];

        Assert.True(missing.Length == 0, $"The bundled FFmpeg libraries for {rid} are missing from the build output: {string.Join(", ", missing)}.");
        Assert.True(BundledFfmpeg.IsAvailable, BundledFfmpeg.UnavailableReason);
    }

    [Fact]
    public void TheBuildIsLgplAndMatchesTheBindings()
    {
        Assert.True(BundledFfmpeg.IsAvailable, BundledFfmpeg.UnavailableReason);

        Assert.StartsWith("LGPL version 2.1 or later", BundledFfmpeg.License, StringComparison.Ordinal);
        Assert.StartsWith("9.0", BundledFfmpeg.Version, StringComparison.Ordinal);
        Assert.Equal(ffmpeg.LibraryVersionMap["avcodec"], (int)(ffmpeg.avcodec_version() >> 16));
    }

    [Fact]
    public void TheConfiguredEncodersAndDecodersExist()
    {
        Assert.True(BundledFfmpeg.IsAvailable, BundledFfmpeg.UnavailableReason);

        string[] encoders = ["libopenh264", "libsvtav1", "libvpx-vp9", "libvpx", "prores_ks", "ffv1", "apng", "gif", "aac", "libopus", "flac", "pcm_s16le"];
        string[] decoders = ["h264", "hevc", "vp8", "vp9", "libdav1d", "prores", "mpeg4", "aac", "opus", "mp3float"];
        Assert.All(encoders, name => Assert.True(Has(name, encoder: true), $"encoder {name}"));
        Assert.All(decoders, name => Assert.True(Has(name, encoder: false), $"decoder {name}"));
        Assert.False(Has("libx264", encoder: true));
    }

    [Fact]
    public void LicenseTextsShipNextToTheLibraries()
    {
        string licenses = Path.Combine(AppContext.BaseDirectory, "licenses", "ffmpeg");

        Assert.True(File.Exists(Path.Combine(licenses, "FFmpeg-COPYING.LGPLv2.1.txt")));
        Assert.True(File.Exists(Path.Combine(licenses, "THIRD-PARTY-NOTICES.md")));
    }

    private static unsafe bool Has(string name, bool encoder)
    {
        return encoder ? ffmpeg.avcodec_find_encoder_by_name(name) is not null : ffmpeg.avcodec_find_decoder_by_name(name) is not null;
    }
}
