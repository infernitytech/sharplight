using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;

namespace Sharplight.Hub.Tests;

/// <summary>A repository: files by path; files of 1 KiB or more are served as LFS files with a SHA-256.</summary>
internal sealed class FakeRepository(string id, string commit, bool gated)
{
    public const int LfsThreshold = 1024;

    public string Id { get; } = id;

    public string Commit { get; } = commit;

    public bool Gated { get; } = gated;

    public Dictionary<string, byte[]> Files { get; } = new(StringComparer.Ordinal);

    /// <summary>Overrides the SHA-256 the tree reports for a file (checksum mismatch tests).</summary>
    public Dictionary<string, string> ReportedSha256 { get; } = new(StringComparer.Ordinal);

    public FakeRepository With(string path, byte[] content)
    {
        Files[path] = content;
        return this;
    }

    public FakeRepository With(string path, string text) => With(path, Encoding.UTF8.GetBytes(text));

    public FakeRepository WithSize(string path, int size) => With(path, Content(path, size));

    public string TreeJson()
    {
        var entries = Files.Select(file =>
        {
            var entry = new JsonObject { ["type"] = "file", ["path"] = file.Key, ["size"] = file.Value.Length };
            if (file.Value.Length >= LfsThreshold)
            {
                var sha = ReportedSha256.TryGetValue(file.Key, out var reported) ? reported : Convert.ToHexStringLower(SHA256.HashData(file.Value));
                entry["lfs"] = new JsonObject { ["oid"] = sha, ["size"] = file.Value.Length };
                entry["oid"] = "ffffffffffffffffffffffffffffffffffffffff";
            }
            else
            {
                entry["oid"] = GitOid(file.Value);
            }

            return (JsonNode)entry;
        });
        return new JsonArray([.. entries]).ToJsonString();
    }

    /// <summary>Deterministic pseudo-random content.</summary>
    public static byte[] Content(string seed, int size)
    {
        var bytes = new byte[size];
        new Random(seed.GetHashCode(StringComparison.Ordinal)).NextBytes(bytes);
        return bytes;
    }

    private static string GitOid(byte[] content)
    {
        var header = Encoding.ASCII.GetBytes(string.Create(CultureInfo.InvariantCulture, $"blob {content.Length}\0"));
        return Convert.ToHexStringLower(SHA1.HashData([.. header, .. content]));
    }
}
