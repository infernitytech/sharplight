using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace Sharplight.Shaders;

/// <summary>The embedded Slang kernel sources, extracted to a content-addressed directory on first use.</summary>
internal static partial class KernelLibrary
{
    private const string Prefix = "Sharplight.Kernels.";
    private static readonly Lazy<Extracted> s_extracted = new(DoExtract);
    private static readonly ConcurrentDictionary<string, string> s_closureHashes = new(StringComparer.Ordinal);

    /// <summary>Directory holding the extracted kernel sources.</summary>
    public static string Extract() => s_extracted.Value.Directory;

    /// <summary>Text of a library file (read once: the files are embedded and never change during a process).</summary>
    public static string ReadSource(string file)
        => s_extracted.Value.Sources.TryGetValue(file, out var text) ? text : File.ReadAllText(Path.Combine(Extract(), file));

    /// <summary>
    /// Content hash of every library file that <paramref name="source"/> includes, directly or transitively, so that a kernel's
    /// cache key changes exactly when something it compiles against changes.
    /// </summary>
    /// <remarks>Only <c>#include "file"</c> directives pull library files in; the macros do not select other files.</remarks>
    public static string IncludeHash(string source)
    {
        var names = new SortedSet<string>(StringComparer.Ordinal);
        foreach (Match match in IncludeDirective().Matches(source))
        {
            names.Add(match.Groups[1].Value);
        }

        return string.Join(',', names.Select(ClosureHash));
    }

    [GeneratedRegex(@"^\s*#\s*include\s+""([^""]+)""", RegexOptions.Multiline)]
    private static partial Regex IncludeDirective();

    /// <summary>Hash of a file's text combined with the closure hashes of the files it includes.</summary>
    private static string ClosureHash(string file)
    {
        if (s_closureHashes.TryGetValue(file, out var known))
        {
            return known;
        }

        if (!s_extracted.Value.Sources.TryGetValue(file, out var text))
        {
            return file + ":missing";
        }

        var sb = new StringBuilder(text);
        foreach (Match match in IncludeDirective().Matches(text))
        {
            sb.Append('\n').Append(ClosureHash(match.Groups[1].Value));
        }

        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString())))[..16].ToLowerInvariant();
        return s_closureHashes[file] = hash;
    }

    private static Extracted DoExtract()
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
        var sources = files.ToDictionary(kv => kv.Key, kv => Encoding.UTF8.GetString(kv.Value), StringComparer.Ordinal);

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
        return new Extracted(dir, sources);
    }

    private sealed record Extracted(string Directory, Dictionary<string, string> Sources);
}
