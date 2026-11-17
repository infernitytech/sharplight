using System.Security.Cryptography;

namespace Sharplight.ModelFeed.Signing;

/// <summary>
/// A public ECDSA P-384 key that verifies feed signatures (ES384, SHA-384, IEEE P1363 signature format as in JWS). Its id is
/// <c>sha256:</c> plus the first 32 hex digits of the SHA-256 of the SubjectPublicKeyInfo. Immutable and thread safe.
/// </summary>
public sealed class FeedPublicKey
{
    /// <summary>The JWS algorithm name.</summary>
    public const string Algorithm = "ES384";

    private readonly byte[] _subjectPublicKeyInfo;

    private FeedPublicKey(byte[] subjectPublicKeyInfo)
    {
        using var key = ECDsa.Create();
        key.ImportSubjectPublicKeyInfo(subjectPublicKeyInfo, out _);
        if (key.KeySize != 384)
        {
            throw new FeedFormatException($"Feed keys are P-384; this key has {key.KeySize} bits.");
        }

        _subjectPublicKeyInfo = subjectPublicKeyInfo;
        KeyId = IdOf(subjectPublicKeyInfo);
    }

    /// <summary>The key id.</summary>
    public string KeyId { get; }

    /// <summary>The SubjectPublicKeyInfo, base64.</summary>
    public string Base64 => Convert.ToBase64String(_subjectPublicKeyInfo);

    /// <summary>Reads a key from base64 SubjectPublicKeyInfo, a PEM <c>PUBLIC KEY</c> block, or the JSON text of a key file.</summary>
    /// <param name="text">The key text.</param>
    /// <returns>The key.</returns>
    /// <exception cref="FeedFormatException">The text is not a P-384 public key.</exception>
    public static FeedPublicKey Parse(string text)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        var trimmed = text.Trim();
        try
        {
            if (trimmed.StartsWith('{'))
            {
                var file = FeedJson.Read(System.Text.Encoding.UTF8.GetBytes(trimmed), FeedJsonContext.Default.FeedKeyFile, "The key file");
                return FromBase64(file.PublicKey);
            }

            if (trimmed.StartsWith("-----", StringComparison.Ordinal))
            {
                using var key = ECDsa.Create();
                key.ImportFromPem(trimmed);
                return new FeedPublicKey(key.ExportSubjectPublicKeyInfo());
            }

            return FromBase64(trimmed);
        }
        catch (Exception exception) when (exception is CryptographicException or FormatException or ArgumentException)
        {
            throw new FeedFormatException($"Not a P-384 public key: {exception.Message}", exception);
        }
    }

    /// <summary>Reads a key from base64 SubjectPublicKeyInfo.</summary>
    /// <param name="base64">The key.</param>
    /// <returns>The key.</returns>
    /// <exception cref="FeedFormatException">The text is not a P-384 public key.</exception>
    public static FeedPublicKey FromBase64(string base64)
    {
        try
        {
            return new FeedPublicKey(Convert.FromBase64String(base64));
        }
        catch (Exception exception) when (exception is CryptographicException or FormatException)
        {
            throw new FeedFormatException($"Not a P-384 public key: {exception.Message}", exception);
        }
    }

    /// <summary>Loads a public or private key file and returns its public key.</summary>
    /// <param name="path">Path of the JSON key file (or a PEM file).</param>
    /// <returns>The key.</returns>
    public static FeedPublicKey Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>Verifies an ES384 signature.</summary>
    /// <param name="data">Signed bytes.</param>
    /// <param name="signature">Signature (r || s, 96 bytes).</param>
    /// <returns><see langword="true"/> when valid.</returns>
    public bool Verify(ReadOnlySpan<byte> data, ReadOnlySpan<byte> signature)
    {
        using var key = ECDsa.Create();
        key.ImportSubjectPublicKeyInfo(_subjectPublicKeyInfo, out _);
        return key.VerifyData(data, signature, HashAlgorithmName.SHA384, DSASignatureFormat.IeeeP1363FixedFieldConcatenation);
    }

    /// <inheritdoc />
    public override string ToString() => KeyId;

    internal static string IdOf(ReadOnlySpan<byte> subjectPublicKeyInfo) =>
        "sha256:" + Convert.ToHexStringLower(SHA256.HashData(subjectPublicKeyInfo))[..32];
}
