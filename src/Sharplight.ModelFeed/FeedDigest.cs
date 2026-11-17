using System.Security.Cryptography;

namespace Sharplight.ModelFeed;

/// <summary>Digests of feed documents: <c>sha256:&lt;lower-case hex&gt;</c> over the exact bytes.</summary>
public static class FeedDigest
{
    private const string Prefix = "sha256:";

    /// <summary>The digest of <paramref name="bytes"/>.</summary>
    /// <param name="bytes">Document bytes.</param>
    /// <returns>The digest text.</returns>
    public static string Of(ReadOnlySpan<byte> bytes) => Prefix + Convert.ToHexStringLower(SHA256.HashData(bytes));

    /// <summary>Whether <paramref name="bytes"/> have the digest <paramref name="digest"/>.</summary>
    /// <param name="bytes">Document bytes.</param>
    /// <param name="digest">Expected digest.</param>
    /// <returns><see langword="true"/> when equal.</returns>
    public static bool Matches(ReadOnlySpan<byte> bytes, string? digest) =>
        digest is not null && string.Equals(Of(bytes), digest.Trim(), StringComparison.OrdinalIgnoreCase);

    /// <summary>Throws unless <paramref name="bytes"/> have the digest <paramref name="digest"/>.</summary>
    /// <param name="bytes">Document bytes.</param>
    /// <param name="digest">Expected digest.</param>
    /// <param name="what">The document, for the message.</param>
    /// <exception cref="FeedVerificationException">The digest differs.</exception>
    public static void Require(ReadOnlySpan<byte> bytes, string? digest, string what)
    {
        if (!Matches(bytes, digest))
        {
            throw new FeedVerificationException($"{what} does not match its digest in the feed (tampered or inconsistent).");
        }
    }
}
