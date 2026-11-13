using System.Security.Cryptography;
using System.Text;

namespace Sharplight.Shaders;

/// <summary>The embedded Slang kernel sources, extracted to a content-addressed directory on first use.</summary>
public static class KernelLibrary
{
    private const string Prefix = "Sharplight.Kernels.";
    private static readonly Lazy<(string Dir, string Hash)> s_extracted = new(DoExtract);

    public static string Hash => s_extracted.Value.Hash;
    public static string Extract() => s_extracted.Value.Dir;

    public static IEnumerable<string> Files =>
        typeof(KernelLibrary).Assembly.GetManifestResourceNames().Where(n => n.StartsWith(Prefix, StringComparison.Ordinal)).Select(n => n[Prefix.Length..]);

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
        foreach (var (k, v) in files) { sha.AppendData(Encoding.UTF8.GetBytes(k)); sha.AppendData(v); }
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

            try { Directory.Move(tmp, dir); }
            catch (IOException) { Directory.Delete(tmp, true); } // another process won the race
        }
        return (dir, hash);
    }
}
