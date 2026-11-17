using Sharplight.ModelFeed.Signing;

namespace Sharplight.ModelFeed.Tests;

public sealed class SignatureTests
{
    [Fact]
    public void DetachedSignature_RoundTrips_AndRejectsChanges()
    {
        using var key = FeedSigningKey.Generate();
        var document = "{\"a\":1}\n"u8.ToArray();
        var signature = key.SignDocument(document, "test");
        var verifiedBy = DetachedSignature.Verify(document, signature, kid => kid == key.KeyId ? key.PublicKey : null, "doc");
        Assert.Equal(key.KeyId, verifiedBy.KeyId);

        var changed = "{\"a\":2}\n"u8.ToArray();
        Assert.Throws<FeedVerificationException>(() => DetachedSignature.Verify(changed, signature, kid => key.PublicKey, "doc"));
        using var other = FeedSigningKey.Generate();
        Assert.Throws<FeedVerificationException>(() => DetachedSignature.Verify(document, signature, kid => null, "doc"));
        Assert.Throws<FeedVerificationException>(() => DetachedSignature.Verify(document, signature, kid => other.PublicKey, "doc"));
    }

    [Fact]
    public void KeyFiles_RoundTrip()
    {
        var folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "modelfeed-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            using var key = FeedSigningKey.Generate("unit test");
            var privatePath = Path.Combine(folder, "feed.key.json");
            var publicPath = Path.Combine(folder, "feed.pub.json");
            key.Save(privatePath, publicPath);
            using var loaded = FeedSigningKey.Load(privatePath);
            Assert.Equal(key.KeyId, loaded.KeyId);
            Assert.Equal(key.KeyId, FeedPublicKey.Load(publicPath).KeyId);
            Assert.Throws<FeedFormatException>(() => FeedSigningKey.Load(publicPath));
            Assert.Equal(key.KeyId, FeedPublicKey.Parse(key.PublicKey.Base64).KeyId);
            if (!OperatingSystem.IsWindows() && FileModesWork(folder))
            {
                Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite, File.GetUnixFileMode(privatePath));
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void PemKeys_AreAccepted()
    {
        using var ecdsa = System.Security.Cryptography.ECDsa.Create(System.Security.Cryptography.ECCurve.NamedCurves.nistP384);
        using var key = FeedSigningKey.Parse(ecdsa.ExportPkcs8PrivateKeyPem());
        var publicKey = FeedPublicKey.Parse(ecdsa.ExportSubjectPublicKeyInfoPem());
        Assert.Equal(publicKey.KeyId, key.KeyId);
        Assert.Equal(key.KeyId, FeedPublicKey.Parse(ecdsa.ExportECPrivateKeyPem()).KeyId);
        var document = "{}"u8.ToArray();
        DetachedSignature.Verify(document, key.SignDocument(document, "test"), kid => kid == publicKey.KeyId ? publicKey : null, "doc");
        Assert.Throws<FeedFormatException>(() => FeedSigningKey.Parse(ecdsa.ExportSubjectPublicKeyInfoPem()));
        using var p256 = System.Security.Cryptography.ECDsa.Create(System.Security.Cryptography.ECCurve.NamedCurves.nistP256);
        Assert.Throws<FeedFormatException>(() => FeedSigningKey.Parse(p256.ExportPkcs8PrivateKeyPem()));
    }

    [System.Runtime.Versioning.UnsupportedOSPlatform("windows")]
    private static bool FileModesWork(string folder)
    {
        // Some network file systems ignore chmod; the key file's mode can only be checked where modes are kept.
        var probe = Path.Combine(folder, "probe");
        File.WriteAllText(probe, "");
        File.SetUnixFileMode(probe, UnixFileMode.UserRead | UnixFileMode.UserWrite);
        return File.GetUnixFileMode(probe) == (UnixFileMode.UserRead | UnixFileMode.UserWrite);
    }
}
