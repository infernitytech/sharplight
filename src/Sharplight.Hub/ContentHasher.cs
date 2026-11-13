using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace Sharplight.Hub;

/// <summary>
/// Hashes a download incrementally for verification: SHA-256 of the content for LFS files, the git blob id
/// (SHA-1 over <c>"blob {size}\0"</c> and the content) for files stored in git, or nothing when no checksum is known.
/// </summary>
internal sealed class ContentHasher : IDisposable
{
    private readonly IncrementalHash? _hash;
    private readonly byte[] _prefix;

    private ContentHasher(IncrementalHash? hash, byte[] prefix, string? expected)
    {
        _hash = hash;
        _prefix = prefix;
        Expected = expected;
        Reset();
    }

    /// <summary>The expected lower-case hex digest, or null when nothing is checked.</summary>
    public string? Expected { get; }

    public static ContentHasher For(DownloadRequest request)
    {
        if (request.Sha256 is { } sha256)
        {
            return new ContentHasher(IncrementalHash.CreateHash(HashAlgorithmName.SHA256), [], sha256.ToLowerInvariant());
        }

        if (request.GitOid is { } oid)
        {
            var prefix = Encoding.ASCII.GetBytes(string.Create(CultureInfo.InvariantCulture, $"blob {request.Size}\0"));
            return new ContentHasher(IncrementalHash.CreateHash(HashAlgorithmName.SHA1), prefix, oid.ToLowerInvariant());
        }

        return new ContentHasher(null, [], null);
    }

    public void Append(ReadOnlySpan<byte> data) => _hash?.AppendData(data);

    /// <summary>Starts over (the download restarts from the first byte).</summary>
    public void Reset()
    {
        if (_hash is null)
        {
            return;
        }

        _hash.GetHashAndReset();
        _hash.AppendData(_prefix);
    }

    /// <summary>The digest of everything appended, as lower-case hex; null when nothing is checked.</summary>
    public string? Finish() => _hash is null ? null : Convert.ToHexStringLower(_hash.GetHashAndReset());

    public void Dispose() => _hash?.Dispose();
}
