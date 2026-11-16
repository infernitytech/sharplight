using System.Diagnostics.CodeAnalysis;

namespace Sharplight.Video;

/// <summary>Finds the registered <see cref="IVideoEncoder"/> for a format or a file name; resolve it from the service provider.</summary>
public sealed class VideoEncoderCatalog
{
    private readonly IVideoEncoder[] _encoders;

    /// <summary>Creates the catalog over the registered encoders.</summary>
    /// <param name="encoders">All registered encoders; for a format with several, the first available one wins.</param>
    public VideoEncoderCatalog(IEnumerable<IVideoEncoder> encoders)
    {
        ArgumentNullException.ThrowIfNull(encoders);
        _encoders = [.. encoders];
    }

    /// <summary>The encoders that are usable on this machine, one per format.</summary>
    public IReadOnlyList<IVideoEncoder> Available => [.. _encoders.Where(e => e.IsAvailable).DistinctBy(e => e.Format, StringComparer.OrdinalIgnoreCase)];

    /// <summary>Finds the available encoder of a format.</summary>
    /// <param name="format">The format id (case-insensitive), for example <see cref="VideoFormats.WebM"/>.</param>
    /// <param name="encoder">The encoder when found.</param>
    /// <returns>Whether an available encoder exists.</returns>
    public bool TryFind(string format, [NotNullWhen(true)] out IVideoEncoder? encoder)
    {
        ArgumentException.ThrowIfNullOrEmpty(format);
        return Pick(e => string.Equals(e.Format, format, StringComparison.OrdinalIgnoreCase), out encoder);
    }

    /// <summary>Finds the available encoder for a file name by its extension.</summary>
    /// <param name="path">A file name or path.</param>
    /// <param name="encoder">The encoder when found.</param>
    /// <returns>Whether an available encoder handles the extension.</returns>
    public bool TryFindForPath(string path, [NotNullWhen(true)] out IVideoEncoder? encoder)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string extension = Path.GetExtension(path);
        return Pick(e => e.FileExtensions.Contains(extension, StringComparer.OrdinalIgnoreCase), out encoder);
    }

    /// <summary>Finds the available encoder for a file name by its extension.</summary>
    /// <param name="path">A file name or path.</param>
    /// <returns>The encoder, or <see langword="null"/> when none is available for the extension.</returns>
    public IVideoEncoder? FindForPath(string path)
    {
        return TryFindForPath(path, out IVideoEncoder? encoder) ? encoder : null;
    }

    /// <summary>Encodes the clip to a file, choosing the format by the file extension.</summary>
    /// <param name="clip">The clip.</param>
    /// <param name="path">The file (created or overwritten); its extension selects the format.</param>
    /// <param name="options">Encode settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the encode; a partial file is deleted.</param>
    /// <exception cref="NotSupportedException">No available encoder handles the extension.</exception>
    public async Task SaveAsync(VideoClip clip, string path, VideoEncodeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(clip);
        if (!TryFindForPath(path, out IVideoEncoder? encoder))
        {
            string known = string.Join(", ", Available.SelectMany(e => e.FileExtensions));
            throw new NotSupportedException($"No available video encoder for '{Path.GetExtension(path)}' (available: {known}).");
        }

        bool completed = false;
        try
        {
            await using var file = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16, useAsync: true);
            await encoder.EncodeAsync(clip, file, options, cancellationToken).ConfigureAwait(false);
            completed = true;
        }
        finally
        {
            if (!completed)
            {
                File.Delete(path);
            }
        }
    }

    private bool Pick(Func<IVideoEncoder, bool> matches, [NotNullWhen(true)] out IVideoEncoder? encoder)
    {
        foreach (IVideoEncoder candidate in _encoders)
        {
            if (candidate.IsAvailable && matches(candidate))
            {
                encoder = candidate;
                return true;
            }
        }

        encoder = null;
        return false;
    }
}
