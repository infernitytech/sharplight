using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.Options;

namespace Sharplight.Video;

/// <summary>
/// MP4 with H.264 (<see cref="VideoFormats.Mp4"/>) through a system <c>ffmpeg</c> that must be installed separately (H.264 is patent
/// encumbered, so no encoder is bundled and ffmpeg is only ever run as a separate process). Raw RGB frames are piped to
/// <c>ffmpeg -f rawvideo -pix_fmt rgb24 ... -c:v libx264 -pix_fmt yuv420p</c>, which writes a fragmented MP4 (<c>frag_keyframe+empty_moov</c>:
/// the movie header first, so no seekable output and no temporary file are needed) to its standard output, copied to the destination
/// stream as it arrives. <see cref="IsAvailable"/> is false when ffmpeg is not found.
/// </summary>
public sealed class FfmpegVideoEncoder : IVideoEncoder
{
    private readonly SharplightVideoOptions _options;
    private readonly Lazy<string?> _executable;

    /// <summary>Creates the encoder.</summary>
    /// <param name="options">The ffmpeg path; <see langword="null"/> searches <c>PATH</c>.</param>
    public FfmpegVideoEncoder(IOptions<SharplightVideoOptions>? options = null)
    {
        _options = options?.Value ?? new SharplightVideoOptions();
        _executable = new Lazy<string?>(() => FfmpegLocator.Find(_options.FfmpegPath));
    }

    /// <inheritdoc />
    public string Format => VideoFormats.Mp4;

    /// <inheritdoc />
    public IReadOnlyList<string> FileExtensions { get; } = [".mp4", ".m4v"];

    /// <inheritdoc />
    public bool IsAvailable => _executable.Value is not null;

    /// <inheritdoc />
    public async Task EncodeAsync(VideoClip clip, Stream output, VideoEncodeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(output);
        string executable = _executable.Value ?? throw new PlatformNotSupportedException("ffmpeg was not found; install it or set SharplightVideoOptions.FfmpegPath to write MP4 files.");
        VideoEncodeOptions settings = options ?? new VideoEncodeOptions();
        await RunAsync(executable, clip, settings, output, cancellationToken).ConfigureAwait(false);
    }

    private static async Task RunAsync(string executable, VideoClip clip, VideoEncodeOptions options, Stream output, CancellationToken cancellationToken)
    {
        var start = new ProcessStartInfo(executable)
        {
            RedirectStandardInput = true,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (string argument in Arguments(clip, options))
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("ffmpeg could not be started.");
        using CancellationTokenRegistration registration = cancellationToken.Register(() => Kill(process));
        Task<string> errors = process.StandardError.ReadToEndAsync(CancellationToken.None);
        Task copy = process.StandardOutput.BaseStream.CopyToAsync(output, cancellationToken);
        try
        {
            Stream input = process.StandardInput.BaseStream;
            foreach (var frame in clip.Frames)
            {
                await input.WriteAsync(frame.Rgb, cancellationToken).ConfigureAwait(false);
            }

            await input.FlushAsync(cancellationToken).ConfigureAwait(false);
            input.Close();
        }
        catch (IOException) when (process.HasExited)
        {
            // ffmpeg stopped reading because it failed; the exit code and its message below explain why.
        }

        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        string message = await errors.ConfigureAwait(false);
        await copy.ConfigureAwait(false);
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"ffmpeg failed with exit code {process.ExitCode}: {message.Trim()}");
        }
    }

    private static IEnumerable<string> Arguments(VideoClip clip, VideoEncodeOptions options)
    {
        string size = string.Create(CultureInfo.InvariantCulture, $"{clip.Width}x{clip.Height}");
        string rate = clip.FramesPerSecond.ToString("R", CultureInfo.InvariantCulture);
        string[] input = ["-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", rate, "-i", "-"];
        foreach (string argument in input)
        {
            yield return argument;
        }

        // H.264 in 4:2:0 needs even dimensions: pad odd sizes by one pixel.
        yield return "-vf";
        yield return "pad=ceil(iw/2)*2:ceil(ih/2)*2";
        yield return "-c:v";
        yield return "libx264";
        yield return "-pix_fmt";
        yield return "yuv420p";
        if (options.BitrateKilobitsPerSecond is int bitrate)
        {
            yield return "-b:v";
            yield return string.Create(CultureInfo.InvariantCulture, $"{bitrate}k");
        }
        else
        {
            yield return "-crf";
            yield return string.Create(CultureInfo.InvariantCulture, $"{40 - Math.Clamp(options.Quality, 0, 100) * 26 / 100}");
        }

        if (options.MaxThreads > 0)
        {
            yield return "-threads";
            yield return options.MaxThreads.ToString(CultureInfo.InvariantCulture);
        }

        yield return "-movflags";
        yield return "frag_keyframe+empty_moov+default_base_moof";
        yield return "-f";
        yield return "mp4";
        yield return "pipe:1";
    }

    private static void Kill(Process process)
    {
        try
        {
            process.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException)
        {
            // The process has already exited.
        }
    }
}
