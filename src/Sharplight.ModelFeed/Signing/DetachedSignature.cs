using System.Buffers.Text;
using System.Text;

namespace Sharplight.ModelFeed.Signing;

/// <summary>
/// Detached JWS compact signatures (RFC 7515 appendix F): <c>BASE64URL(header)..BASE64URL(signature)</c> over
/// <c>BASE64URL(header) '.' BASE64URL(document)</c>, algorithm ES384. The document itself is not embedded, so the signed file stays
/// readable and its bytes are exactly what is hashed.
/// </summary>
public static class DetachedSignature
{
    /// <summary>Signs a document.</summary>
    /// <param name="document">Document bytes.</param>
    /// <param name="key">Signing key.</param>
    /// <param name="type">Content type for the header, or <see langword="null"/>.</param>
    /// <returns>The compact detached JWS.</returns>
    public static string Create(ReadOnlySpan<byte> document, FeedSigningKey key, string? type)
    {
        ArgumentNullException.ThrowIfNull(key);
        var header = FeedJson.Write(new JwsHeader { Alg = FeedPublicKey.Algorithm, Kid = key.KeyId, Typ = type }, FeedJsonContext.Default.JwsHeader);
        var encodedHeader = Base64Url.EncodeToString(header.AsSpan(0, header.Length - 1));
        var signature = key.Sign(SigningInput(encodedHeader, document));
        return $"{encodedHeader}..{Base64Url.EncodeToString(signature)}";
    }

    /// <summary>Reads the key id of a signature without verifying it.</summary>
    /// <param name="signature">The compact detached JWS (UTF-8 bytes, surrounding whitespace allowed).</param>
    /// <returns>The key id.</returns>
    /// <exception cref="FeedVerificationException">The signature is malformed.</exception>
    public static string ReadKeyId(ReadOnlySpan<byte> signature) => Split(signature).Header.Kid;

    /// <summary>Verifies a signature with one of <paramref name="keys"/>.</summary>
    /// <param name="document">Document bytes.</param>
    /// <param name="signature">The compact detached JWS (UTF-8 bytes).</param>
    /// <param name="keys">Resolves a key id to a trusted key, or <see langword="null"/> when it is not trusted.</param>
    /// <param name="what">The document, for messages.</param>
    /// <returns>The key that verified the signature.</returns>
    /// <exception cref="FeedVerificationException">The signature is malformed, the key is not trusted, or the signature is invalid.</exception>
    public static FeedPublicKey Verify(ReadOnlySpan<byte> document, ReadOnlySpan<byte> signature, Func<string, FeedPublicKey?> keys, string what)
    {
        ArgumentNullException.ThrowIfNull(keys);
        var (encodedHeader, header, signatureBytes) = Split(signature);
        if (header.Alg != FeedPublicKey.Algorithm)
        {
            throw new FeedVerificationException($"{what}: signature algorithm '{header.Alg}' is not accepted ({FeedPublicKey.Algorithm} is).");
        }

        var key = keys(header.Kid) ?? throw new FeedVerificationException($"{what} is signed by key {header.Kid}, which is not trusted (unknown, expired or revoked).");
        if (!key.Verify(SigningInput(encodedHeader, document), signatureBytes))
        {
            throw new FeedVerificationException($"{what}: the signature is invalid (the document was changed or signed by another key).");
        }

        return key;
    }

    private static byte[] SigningInput(string encodedHeader, ReadOnlySpan<byte> document)
    {
        var encodedDocument = Base64Url.EncodeToUtf8(document);
        var header = Encoding.ASCII.GetBytes(encodedHeader);
        var input = new byte[header.Length + 1 + encodedDocument.Length];
        header.CopyTo(input, 0);
        input[header.Length] = (byte)'.';
        encodedDocument.CopyTo(input, header.Length + 1);
        return input;
    }

    private static (string EncodedHeader, JwsHeader Header, byte[] Signature) Split(ReadOnlySpan<byte> signature)
    {
        var text = Encoding.UTF8.GetString(signature).Trim();
        var parts = text.Split('.');
        if (parts.Length != 3 || parts[1].Length != 0 || parts[0].Length == 0 || parts[2].Length == 0)
        {
            throw new FeedVerificationException("The signature is not a detached JWS (header..signature).");
        }

        try
        {
            var header = FeedJson.Read(Base64Url.DecodeFromChars(parts[0]), FeedJsonContext.Default.JwsHeader, "The signature header");
            return (parts[0], header, Base64Url.DecodeFromChars(parts[2]));
        }
        catch (Exception exception) when (exception is FormatException or FeedFormatException)
        {
            throw new FeedVerificationException($"The signature is malformed: {exception.Message}", exception);
        }
    }
}
