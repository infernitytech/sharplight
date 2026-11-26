using System.Globalization;
using Sharplight.Imaging;
using Sharplight.Samples;
using Sharplight.Video;
using Sharplight.VideoEnhancement;
using Sharplight.Vision;

namespace Sharplight.EnhanceVideo;

/// <summary>
/// Where the enhanced frames go as they finish: a PNG (one frame), a video file encoded incrementally (the encoder chosen by the extension,
/// with the input's sound track), or, without an extension, numbered PNGs in a folder.
/// </summary>
internal sealed class EnhancedFrameOutput(EnhanceVideoSettings settings, VideoEncoderCatalog encoders) : IAsyncDisposable
{
    private IVideoFrameWriter? _writer;
    private long _written;

    /// <summary>Writes a batch of finished frames.</summary>
    /// <param name="frames">The frames, in order.</param>
    /// <param name="update">The update that carried them (size, rate, position).</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task WriteAsync(IReadOnlyList<ImageInput> frames, VideoEnhancementUpdate update, CancellationToken cancellationToken)
    {
        string output = settings.Output;
        string extension = Path.GetExtension(output);
        if (string.Equals(extension, ".png", StringComparison.OrdinalIgnoreCase))
        {
            if (_written > 0 || frames.Count > 1)
            {
                throw new SampleException("A .png output takes one frame; give a video file or a folder for a clip.", ExitCodes.InvalidInput);
            }

            PngCodec.Write(output, frames[0]);
            _written++;
            return;
        }

        if (extension.Length == 0)
        {
            Directory.CreateDirectory(output);
            foreach (var frame in frames)
            {
                PngCodec.Write(Path.Combine(output, string.Create(CultureInfo.InvariantCulture, $"frame-{_written++:D6}.png")), frame);
            }

            return;
        }

        _writer ??= OpenWriter(update);
        foreach (var frame in frames)
        {
            await _writer.WriteAsync(frame, cancellationToken);
            _written++;
        }
    }

    /// <summary>Finishes the output (the video file's trailer) and reports it.</summary>
    /// <param name="cancellationToken">Cancels the finishing.</param>
    public async Task CompleteAsync(CancellationToken cancellationToken)
    {
        if (_writer is not null)
        {
            await _writer.CompleteAsync(cancellationToken);
        }

        Console.WriteLine($"wrote {_written} frame(s) to {settings.Output}");
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync() => _writer?.DisposeAsync() ?? ValueTask.CompletedTask;

    private IVideoFrameWriter OpenWriter(VideoEnhancementUpdate update)
    {
        if (!encoders.TryFindIncrementalForPath(settings.Output, out _))
        {
            throw new SampleException($"No available encoder writes '{Path.GetExtension(settings.Output)}' files frame by frame.", ExitCodes.InvalidInput);
        }

        bool video = !string.Equals(Path.GetExtension(settings.Input), ".png", StringComparison.OrdinalIgnoreCase);
        return encoders.OpenWriter(settings.Output, new VideoStreamSettings
        {
            Width = update.Size.Width,
            Height = update.Size.Height,
            FramesPerSecond = update.FramesPerSecond,
            AudioSourcePath = video ? settings.Input : null,
            AudioSourceStart = TimeSpan.FromSeconds(settings.Start),
        });
    }
}
