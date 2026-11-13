using System.Security.Cryptography;
using System.Text;

namespace Sharplight.Shaders;

/// <summary>The embedded Slang kernel sources, extracted to a content-addressed directory on first use.</summary>
internal static class KernelLibrary
{
    private const string Prefix = "Sharplight.Kernels.";
    private static readonly Lazy<(string Dir, string Hash)> s_extracted = new(DoExtract);

    /// <summary>Content hash of all kernel sources (part of every kernel cache key).</summary>
    public static string Hash => s_extracted.Value.Hash;
    /// <summary>Directory holding the extracted kernel sources.</summary>
    public static string Extract() => s_extracted.Value.Dir;

    private static (string, string) DoExtract()
    {
        var asm = typeof(KernelLibrary).Assembly;
        var files = new SortedDictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (var res in asm.GetManifestResourceNames().Where(n => n.StartsWith(Prefix, StringComparison.Ordinal)))
        {
            using var s = asm.GetManifestResourceStream(res)!;
            using var ms = new MemoryStream();
            s.CopyTo(ms);
            files[res[Prefix.Length..]] = ms.ToArray();
        }
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var (k, v) in files)
        {
            sha.AppendData(Encoding.UTF8.GetBytes(k));
            sha.AppendData(v);
        }
        var hash = Convert.ToHexString(sha.GetHashAndReset())[..16].ToLowerInvariant();

        var dir = Path.Combine(Path.GetTempPath(), "sharplight-kernels-" + hash);
        if (!Directory.Exists(dir))
        {
            var tmp = dir + "." + Environment.ProcessId;
            Directory.CreateDirectory(tmp);
            foreach (var (k, v) in files)
            {
                File.WriteAllBytes(Path.Combine(tmp, k), v);
            }

            try
            {
                Directory.Move(tmp, dir);
            }
            catch (IOException)
            {
                // Another process won the race to create the same content-addressed directory.
                Directory.Delete(tmp, true);
            }
        }
        return (dir, hash);
    }
}
