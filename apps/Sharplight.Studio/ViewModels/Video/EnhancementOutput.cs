using Sharplight.Studio.Services.Imaging;
using Sharplight.Video;
using Sharplight.VideoEnhancement;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>
/// The file an Enhance run writes into the page's output folder as the batches finish: a PNG for an image; for a video, the first of MP4,
/// WebM and Matroska the bundled FFmpeg writes frame by frame, encoded incrementally with the input's sound track (copied when the format
/// takes its codec, else transcoded). Finishing the file early (a cancelled or failed run) leaves a valid, shorter video; disposing it
/// unfinished deletes it.
/// </summary>
/// <param name="input">The run's input.</param>
/// <param name="encoders">The video encoders.</param>
/// <param name="folder">The output folder (created when the first frames arrive).</param>
internal sealed class EnhancementOutput(EnhancementInput input, VideoEncoderCatalog encoders, string folder) : IAsyncDisposable
{
    private static readonly string[] s_formats = [VideoFormats.Mp4, VideoFormats.WebM, VideoFormats.Matroska];

    private IVideoFrameWriter? _writer;

    /// <summary>The file written, or null before the first frames.</summary>
    public string? Path { get; private set; }

    /// <summary>The first enhanced frame (the run's still for the history), or null before the first frames.</summary>
    public ImageInput? FirstFrame { get; private set; }

    /// <summary>The frames written so far.</summary>
    public long FramesWritten { get; private set; }

    /// <summary>Whether the file has been finished.</summary>
    public bool IsComplete { get; private set; }

    /// <summary>Why the result is not written (no format can be written frame by frame on this machine), or null.</summary>
    public string? WriteError { get; private set; }

    /// <summary>Why the sound track was left out (the input's could not be read), or null.</summary>
    public string? AudioError { get; private set; }

    /// <summary>Writes a batch of finished frames (opening the file with the first).</summary>
    /// <param name="frames">The frames, in order.</param>
    /// <param name="update">The update that carried them (size and rate of the result).</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public async Task WriteAsync(IReadOnlyList<ImageInput> frames, VideoEnhancementUpdate update, CancellationToken cancellationToken)
    {
        if (frames.Count == 0)
        {
            return;
        }

        FirstFrame ??= frames[0];
        if (!input.IsVideo)
        {
            Path = System.IO.Path.Combine(CreateFolder(), Stem(update) + ".png");
            await File.WriteAllBytesAsync(Path, ImageFiles.ToPng(frames[0]), cancellationToken).ConfigureAwait(false);
            FramesWritten = 1;
            return;
        }

        if (WriteError is not null)
        {
            return;
        }

        try
        {
            _writer ??= Open(update);
        }
        catch (NotSupportedException unsupported)
        {
            // The run goes on (the comparison still shows it); the page reports that nothing could be saved.
            WriteError = unsupported.Message;
            return;
        }

        foreach (var frame in frames)
        {
            await _writer.WriteAsync(frame, cancellationToken).ConfigureAwait(false);
            FramesWritten++;
        }
    }

    /// <summary>Finishes the file: the video of the frames written so far, with its sound up to their end.</summary>
    /// <param name="cancellationToken">Cancels the finishing.</param>
    /// <returns>The file, or null when no frame was written.</returns>
    public async Task<string?> CompleteAsync(CancellationToken cancellationToken)
    {
        if (_writer is not null && !IsComplete)
        {
            await _writer.CompleteAsync(cancellationToken).ConfigureAwait(false);
        }

        IsComplete = true;
        return FramesWritten > 0 ? Path : null;
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync() => _writer?.DisposeAsync() ?? ValueTask.CompletedTask;

    private IVideoFrameWriter Open(VideoEnhancementUpdate update)
    {
        foreach (string format in s_formats)
        {
            if (encoders.TryFind(format, out var encoder) && encoder.SupportsIncrementalWriting)
            {
                Path = System.IO.Path.Combine(CreateFolder(), Stem(update) + encoder.FileExtensions[0]);
                var stream = new VideoStreamSettings
                {
                    Width = update.Size.Width,
                    Height = update.Size.Height,
                    FramesPerSecond = update.FramesPerSecond,
                    AudioSourcePath = input.Path,
                };
                try
                {
                    return encoders.OpenWriter(Path, stream);
                }
                catch (Exception error) when (error is InvalidDataException or IOException or PlatformNotSupportedException)
                {
                    AudioError = error.Message;
                    return encoders.OpenWriter(Path, stream with { AudioSourcePath = null });
                }
            }
        }

        throw new NotSupportedException("No video encoder can write the result frame by frame (the bundled FFmpeg libraries are missing).");
    }

    private string CreateFolder()
    {
        Directory.CreateDirectory(folder);
        return folder;
    }

    private string Stem(VideoEnhancementUpdate update)
        => $"{System.IO.Path.GetFileNameWithoutExtension(input.Path)}-{update.Size.Width}x{update.Size.Height}-{DateTime.Now:yyyyMMdd-HHmmss}";
}
