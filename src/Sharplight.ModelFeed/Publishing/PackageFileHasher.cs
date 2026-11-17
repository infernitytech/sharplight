using System.Security.Cryptography;

namespace Sharplight.ModelFeed.Publishing;

/// <summary>Describes a package file for its manifest: size, SHA-256 and the SHA-256 of each chunk.</summary>
public static class PackageFileHasher
{
    /// <summary>Default chunk size (64 MiB).</summary>
    public const long DefaultChunkSize = 64L << 20;

    /// <summary>Hashes <paramref name="path"/>.</summary>
    /// <param name="path">The file.</param>
    /// <param name="name">File name for the manifest.</param>
    /// <param name="chunkSize">Chunk size in bytes.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The manifest entry (role <see cref="PackageFileRoles.Package"/>).</returns>
    public static async Task<PackageFile> DescribeAsync(string path, string name, long chunkSize = DefaultChunkSize, CancellationToken cancellationToken = default)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous | FileOptions.SequentialScan);
        return await DescribeAsync(stream, name, chunkSize, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Hashes the rest of <paramref name="stream"/>.</summary>
    /// <param name="stream">The content (read to the end, not disposed).</param>
    /// <param name="name">File name for the manifest.</param>
    /// <param name="chunkSize">Chunk size in bytes.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The manifest entry (role <see cref="PackageFileRoles.Package"/>).</returns>
    public static async Task<PackageFile> DescribeAsync(Stream stream, string name, long chunkSize = DefaultChunkSize, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkSize);
        using var whole = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var chunk = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var chunks = new List<string>();
        var buffer = new byte[1 << 20];
        long size = 0;
        long inChunk = 0;
        while (true)
        {
            var wanted = (int)Math.Min(buffer.Length, chunkSize - inChunk);
            var read = await stream.ReadAsync(buffer.AsMemory(0, wanted), cancellationToken).ConfigureAwait(false);
            if (read == 0)
            {
                break;
            }

            whole.AppendData(buffer, 0, read);
            chunk.AppendData(buffer, 0, read);
            size += read;
            inChunk += read;
            if (inChunk == chunkSize)
            {
                chunks.Add(Convert.ToHexStringLower(chunk.GetHashAndReset()));
                inChunk = 0;
            }
        }

        if (inChunk > 0)
        {
            chunks.Add(Convert.ToHexStringLower(chunk.GetHashAndReset()));
        }

        return new PackageFile
        {
            Name = name,
            Role = PackageFileRoles.Package,
            Size = size,
            Sha256 = Convert.ToHexStringLower(whole.GetHashAndReset()),
            ChunkSize = chunkSize,
            Chunks = chunks,
        };
    }
}
