using System.Security.Cryptography;
using System.Text;

namespace Sharplight.ModelFeed.Signing;

/// <summary>
/// A private ECDSA P-384 key that signs feed documents (a root key signs key sets, a feed key signs indexes and manifests). Keys
/// live outside repositories: in a key file with restricted permissions, a secret store or a KMS. Not thread safe for concurrent
/// disposal; signing is thread safe.
/// </summary>
public sealed class FeedSigningKey : IDisposable
{
    private readonly ECDsa _key;

    private FeedSigningKey(ECDsa key, string? comment)
    {
        _key = key;
        PublicKey = FeedPublicKey.FromBase64(Convert.ToBase64String(key.ExportSubjectPublicKeyInfo()));
        Comment = comment;
    }

    /// <summary>The public half.</summary>
    public FeedPublicKey PublicKey { get; }

    /// <summary>The key id.</summary>
    public string KeyId => PublicKey.KeyId;

    /// <summary>Free text stored in the key file.</summary>
    public string? Comment { get; }

    /// <summary>Generates a new key.</summary>
    /// <param name="comment">Free text for the key file (purpose, custodian).</param>
    /// <returns>The key.</returns>
    public static FeedSigningKey Generate(string? comment = null) => new(ECDsa.Create(ECCurve.NamedCurves.nistP384), comment);

    /// <summary>Loads a private key file (JSON, <see cref="FeedKeyFile"/>).</summary>
    /// <param name="path">Key file path.</param>
    /// <returns>The key.</returns>
    /// <exception cref="FeedFormatException">The file has no private P-384 key.</exception>
    public static FeedSigningKey Load(string path)
    {
        var file = FeedJson.Read(File.ReadAllBytes(path), FeedJsonContext.Default.FeedKeyFile, $"Key file {path}");
        if (string.IsNullOrEmpty(file.PrivateKey))
        {
            throw new FeedFormatException($"{path} holds no private key (it is a public key file).");
        }

        var key = ECDsa.Create();
        try
        {
            key.ImportPkcs8PrivateKey(Convert.FromBase64String(file.PrivateKey), out _);
            if (key.KeySize != 384)
            {
                throw new FeedFormatException($"Feed keys are P-384; {path} has {key.KeySize} bits.");
            }

            return new FeedSigningKey(key, file.Comment);
        }
        catch (Exception exception) when (exception is CryptographicException or FormatException)
        {
            key.Dispose();
            throw new FeedFormatException($"{path} holds no valid private key: {exception.Message}", exception);
        }
        catch
        {
            key.Dispose();
            throw;
        }
    }

    /// <summary>Writes the private key file (created new, owner-only permissions on Unix) and the public key file next to it.</summary>
    /// <param name="privatePath">Private key file path; must not exist.</param>
    /// <param name="publicPath">Public key file path.</param>
    public void Save(string privatePath, string publicPath)
    {
        var file = new FeedKeyFile
        {
            KeyId = KeyId,
            PublicKey = PublicKey.Base64,
            PrivateKey = Convert.ToBase64String(_key.ExportPkcs8PrivateKey()),
            Comment = Comment,
        };
        var options = new FileStreamOptions { Mode = FileMode.CreateNew, Access = FileAccess.Write };
        if (!OperatingSystem.IsWindows())
        {
            options.UnixCreateMode = UnixFileMode.UserRead | UnixFileMode.UserWrite;
        }

        using (var stream = new FileStream(privatePath, options))
        {
            stream.Write(FeedJson.Write(file, FeedJsonContext.Default.FeedKeyFile));
        }

        if (!OperatingSystem.IsWindows())
        {
            // Some file systems ignore the create mode; set it explicitly.
            File.SetUnixFileMode(privatePath, UnixFileMode.UserRead | UnixFileMode.UserWrite);
        }

        File.WriteAllBytes(publicPath, FeedJson.Write(file with { PrivateKey = null }, FeedJsonContext.Default.FeedKeyFile));
    }

    /// <summary>Signs <paramref name="data"/> (ES384, IEEE P1363 format).</summary>
    /// <param name="data">Bytes to sign.</param>
    /// <returns>The signature.</returns>
    public byte[] Sign(ReadOnlySpan<byte> data) =>
        _key.SignData(data, HashAlgorithmName.SHA384, DSASignatureFormat.IeeeP1363FixedFieldConcatenation);

    /// <summary>Signs a document as a detached JWS (<see cref="DetachedSignature"/>).</summary>
    /// <param name="document">Document bytes.</param>
    /// <param name="type">Content type for the header (<c>feed-index</c>, <c>package-manifest</c>, <c>keyset</c>).</param>
    /// <returns>The signature text (UTF-8).</returns>
    public byte[] SignDocument(ReadOnlySpan<byte> document, string type) => Encoding.UTF8.GetBytes(DetachedSignature.Create(document, this, type) + "\n");

    /// <inheritdoc />
    public void Dispose() => _key.Dispose();
}
