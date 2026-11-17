using System.Security.Cryptography;

namespace Sharplight.ModelFeed.Client;

/// <summary>
/// Downloads a package's <c>.slm</c> from one or more sources (feed, mirrors, peers): resumes a partial file after checking its
/// complete chunks, reads the rest with range requests, verifies every chunk against the manifest while writing and the whole
/// file's SHA-256 at the end, and switches to the next source when one fails or sends wrong bytes. The manifest must come from a
/// verified source (<see cref="FeedClient.GetManifestAsync"/>). Thread safe; one download per destination at a time.
/// </summary>
public sealed class PackageDownloader
{
    private const int BufferSize = 1 << 20;

    /// <summary>Suffix of the partial file next to the destination.</summary>
    public const string PartialSuffix = ".part";

    /// <summary>Downloads the package file of <paramref name="manifest"/> to <paramref name="destination"/>.</summary>
    /// <param name="manifest">A verified manifest.</param>
    /// <param name="sources">Sources in order of preference.</param>
    /// <param name="destination">Destination file; an existing file with the right hash is kept.</param>
    /// <param name="progress">Receives the number of bytes present, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancellation (the partial file is kept for resuming).</param>
    /// <returns>The number of bytes transferred in this call.</returns>
    /// <exception cref="FeedVerificationException">Every source failed and the last failure was a hash mismatch.</exception>
    /// <exception cref="IOException">Every source failed.</exception>
    public async Task<long> DownloadAsync(PackageManifest manifest, IReadOnlyList<IFeedTransport> sources, string destination, IProgress<long>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentException.ThrowIfNullOrWhiteSpace(destination);
        var file = manifest.PrimaryFile ?? throw new FeedFormatException($"The manifest of {manifest.Id} {manifest.Version} lists no package file.");
        if (sources.Count == 0)
        {
            throw new ArgumentException("At least one source is needed.", nameof(sources));
        }

        if (File.Exists(destination) && await HashAsync(destination, cancellationToken).ConfigureAwait(false) == file.Sha256.ToLowerInvariant())
        {
            progress?.Report(file.Size);
            return 0;
        }

        var directory = Path.GetDirectoryName(Path.GetFullPath(destination))!;
        Directory.CreateDirectory(directory);
        var partial = destination + PartialSuffix;
        var path = FeedLayout.VersionFile(manifest.Id, manifest.Version, file.Name);
        long transferred = 0;
        Exception? failure = null;
        foreach (var source in sources)
        {
            try
            {
                var offset = await ValidPrefixAsync(partial, file, cancellationToken).ConfigureAwait(false);
                progress?.Report(offset);
                if (offset < file.Size)
                {
                    transferred += await FetchAsync(source, path, partial, offset, file, progress, cancellationToken).ConfigureAwait(false);
                }

                var actual = await HashAsync(partial, cancellationToken).ConfigureAwait(false);
                if (actual != file.Sha256.ToLowerInvariant())
                {
                    File.Delete(partial);
                    throw new FeedVerificationException($"{file.Name} from {source.Location} does not match its SHA-256 in the manifest.");
                }

                File.Move(partial, destination, overwrite: true);
                return transferred;
            }
            catch (Exception exception) when (exception is IOException or FeedVerificationException or HttpRequestException)
            {
                failure = exception;
            }
        }

        throw failure is FeedVerificationException verification
            ? new FeedVerificationException($"{file.Name} could not be downloaded from any source: {verification.Message}", verification)
            : new IOException($"{file.Name} could not be downloaded from any source: {failure?.Message}", failure);
    }

    private static async Task<long> FetchAsync(IFeedTransport source, string path, string partial, long offset, PackageFile file, IProgress<long>? progress, CancellationToken cancellationToken)
    {
        var chunkSize = file.ChunkSize is > 0 && file.Chunks.Count > 0 ? file.ChunkSize.Value : 0;
        await using var input = await source.OpenReadAsync(path, offset, cancellationToken).ConfigureAwait(false);
        await using var output = new FileStream(partial, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, BufferSize, FileOptions.Asynchronous);
        output.SetLength(offset);
        output.Seek(offset, SeekOrigin.Begin);
        using var chunkHash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[BufferSize];
        var position = offset;
        long transferred = 0;
        var chunkStart = offset;
        while (position < file.Size)
        {
            var wanted = (int)Math.Min(buffer.Length, file.Size - position);
            if (chunkSize > 0)
            {
                wanted = (int)Math.Min(wanted, chunkStart + chunkSize - position);
            }

            var read = await input.ReadAsync(buffer.AsMemory(0, wanted), cancellationToken).ConfigureAwait(false);
            if (read == 0)
            {
                throw new IOException($"{source.Location}{path} ended at byte {position} of {file.Size}.");
            }

            chunkHash.AppendData(buffer, 0, read);
            await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            position += read;
            transferred += read;
            if (chunkSize > 0 && (position - chunkStart == chunkSize || position == file.Size))
            {
                var index = (int)(chunkStart / chunkSize);
                var actual = Convert.ToHexStringLower(chunkHash.GetHashAndReset());
                if (index >= file.Chunks.Count || !string.Equals(actual, file.Chunks[index], StringComparison.OrdinalIgnoreCase))
                {
                    output.SetLength(chunkStart);
                    throw new FeedVerificationException($"Chunk {index} of {file.Name} from {source.Location} does not match the manifest.");
                }

                await output.FlushAsync(cancellationToken).ConfigureAwait(false);
                chunkStart = position;
                progress?.Report(position);
            }
            else if (chunkSize == 0)
            {
                progress?.Report(position);
            }
        }

        if (input.CanSeek && input.Length > file.Size)
        {
            throw new FeedVerificationException($"{file.Name} from {source.Location} is larger than the manifest says.");
        }

        return transferred;
    }

    private static async Task<long> ValidPrefixAsync(string partial, PackageFile file, CancellationToken cancellationToken)
    {
        if (!File.Exists(partial))
        {
            return 0;
        }

        var length = new FileInfo(partial).Length;
        if (length > file.Size)
        {
            File.Delete(partial);
            return 0;
        }

        if (file.ChunkSize is not > 0 || file.Chunks.Count == 0)
        {
            return length;
        }

        var chunkSize = file.ChunkSize.Value;
        await using var stream = new FileStream(partial, FileMode.Open, FileAccess.ReadWrite, FileShare.None, BufferSize, FileOptions.Asynchronous | FileOptions.SequentialScan);
        var buffer = new byte[BufferSize];
        long valid = 0;
        for (var index = 0; index < file.Chunks.Count && valid + Math.Min(chunkSize, file.Size - valid) <= length; index++)
        {
            var expected = Math.Min(chunkSize, file.Size - valid);
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            long remaining = expected;
            while (remaining > 0)
            {
                var read = await stream.ReadAsync(buffer.AsMemory(0, (int)Math.Min(buffer.Length, remaining)), cancellationToken).ConfigureAwait(false);
                if (read == 0)
                {
                    break;
                }

                hash.AppendData(buffer, 0, read);
                remaining -= read;
            }

            if (remaining > 0 || !string.Equals(Convert.ToHexStringLower(hash.GetHashAndReset()), file.Chunks[index], StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            valid += expected;
        }

        stream.SetLength(valid);
        return valid;
    }

    private static async Task<string> HashAsync(string path, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize, FileOptions.Asynchronous | FileOptions.SequentialScan);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false));
    }
}
