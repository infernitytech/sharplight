using System.Collections.Concurrent;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using SlangShaderSharp;

namespace Sharplight.Shaders;

/// <summary>Code generation target for kernels.</summary>
public enum ShaderTarget
{
    /// <summary>SPIR-V for Vulkan compute (pointers = physical storage buffers / BDA).</summary>
    SpirV,
    /// <summary>CUDA C++ source (compile with NVRTC).</summary>
    CudaSource,
    /// <summary>PTX assembly.</summary>
    Ptx,
    /// <summary>Metal Shading Language source.</summary>
    Metal,
}

/// <summary>A compiled kernel binary for one target.</summary>
public sealed record CompiledKernel(string Name, string EntryPoint, ShaderTarget Target, byte[] Code, string CacheKey);

/// <summary>
/// Thin, thread-safe wrapper over the Slang compiler that compiles kernels of the embedded kernel
/// library (or generated source) for any backend target, with an on-disk cache keyed by the source
/// text, macros and target. Shared by all providers (Vulkan, CUDA, Metal).
/// </summary>
public sealed class SlangCompiler : IDisposable
{
    static readonly Lock s_globalLock = new();
    static readonly string? s_dumpDir = Environment.GetEnvironmentVariable("SHARPLIGHT_DUMP_KERNELS");
    static IGlobalSession? s_global;

    readonly string _libraryDir;
    readonly string? _cacheDir;
    readonly ConcurrentDictionary<string, CompiledKernel> _memCache = new();
    // Every instance shares the process-wide Slang global session, which is not thread-safe: compile serially
    // across all instances (a per-instance lock let two devices compile concurrently).
    static readonly Lock _compileLock = new();

    public string LibraryDirectory => _libraryDir;

    /// <param name="cacheDirectory">Directory for compiled binaries (null disables the disk cache).</param>
    public SlangCompiler(string? cacheDirectory = null)
    {
        _cacheDir = cacheDirectory;
        if (_cacheDir is not null) Directory.CreateDirectory(_cacheDir);
        _libraryDir = KernelLibrary.Extract();
    }

    static IGlobalSession Global
    {
        get
        {
            lock (s_globalLock)
            {
                if (s_global is null)
                {
                    var r = Slang.CreateGlobalSession(Slang.ApiVersion, out var gs);
                    if (r.Failed || gs is null) throw new InvalidOperationException($"Failed to create Slang global session ({r}).");
                    s_global = gs;
                }
                return s_global;
            }
        }
    }

    public static string Version => Global.GetBuildTagString();

    public static string DefaultCacheDirectory =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData, Environment.SpecialFolderOption.Create), "sharplight", "kernels");

    /// <summary>Compiles an entry point of a library kernel file (e.g. "matmul.slang", "matvec").</summary>
    public CompiledKernel CompileLibrary(string file, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros,
        Runtime.ICompilationCache? cache = null)
    {
        var source = File.ReadAllText(Path.Combine(_libraryDir, file));
        // Files with several entry points guard each one with #ifdef KERNEL_<entry>.
        var m = new Dictionary<string, string>(macros) { ["KERNEL_" + entryPoint.ToUpperInvariant()] = "1" };
        return Compile(file, source, entryPoint, target, m, cache);
    }

    /// <summary>Compiles generated source (may #include library files).</summary>
    /// <param name="cache">Optional model-level cache (saved compiled models): consulted first, and given every
    /// kernel resolved so it can be exported.</param>
    public CompiledKernel Compile(string name, string source, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros,
        Runtime.ICompilationCache? cache = null)
    {
        var key = CacheKey(source, entryPoint, target, macros);
        string ckey = "kernel/" + key;
        if (s_dumpDir is not null)
        {
            // Diagnostics: SHARPLIGHT_DUMP_KERNELS=<dir> writes every kernel's source and macros.
            Directory.CreateDirectory(s_dumpDir);
            File.WriteAllText(Path.Combine(s_dumpDir, $"{Path.GetFileNameWithoutExtension(name)}_{entryPoint}_{key[..12]}.slang"),
                $"// macros: {FormatMacros(macros)}\n{source}");
        }
        if (_memCache.TryGetValue(key, out var hit))
        {
            if (cache is not null && !cache.TryGet(ckey, out _)) cache.Put(ckey, hit.Code);
            return hit;
        }
        if (cache is not null && cache.TryGet(ckey, out var stored))
            return _memCache[key] = new CompiledKernel(name, entryPoint, target, stored.ToArray(), key);
        var result = CompileNoModelCache(name, source, entryPoint, target, macros, key);
        cache?.Put(ckey, result.Code);
        return result;
    }

    CompiledKernel CompileNoModelCache(string name, string source, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros, string key)
    {

        string? cachePath = _cacheDir is null ? null : Path.Combine(_cacheDir, key + Extension(target));
        // The disk cache is shared by processes; an entry may disappear between check and read (pruned by another
        // process), which is just a miss.
        if (cachePath is not null && TryRead(cachePath) is { } cached)
        {
            var k = new CompiledKernel(name, entryPoint, target, cached, key);
            _memCache[key] = k;
            return k;
        }

        byte[] code;
        lock (_compileLock) // Slang sessions are not thread-safe; compile serially.
            code = CompileUncached(name, source, entryPoint, target, macros);

        if (cachePath is not null)
        {
            var tmp = cachePath + "." + Environment.ProcessId + ".tmp";
            File.WriteAllBytes(tmp, code);
            File.Move(tmp, cachePath, overwrite: true);
        }
        var compiled = new CompiledKernel(name, entryPoint, target, code, key);
        _memCache[key] = compiled;
        return compiled;
    }

    static byte[]? TryRead(string path)
    {
        try { return File.Exists(path) ? File.ReadAllBytes(path) : null; }
        catch (Exception e) when (e is FileNotFoundException or DirectoryNotFoundException or IOException) { return null; }
    }

    byte[] CompileUncached(string name, string source, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros)
    {
        var gs = Global;
        var desc = new SessionDesc
        {
            Targets = [new TargetDesc { Format = Format(target), Profile = gs.FindProfile(Profile(target)) }],
            SearchPaths = [_libraryDir],
            PreprocessorMacros = [.. macros.Select(kv => new PreprocessorMacroDesc(kv.Key, kv.Value))],
        };
        Check(gs.CreateSession(in desc, out var session), "create session", null);
        var module = session.LoadModuleFromSourceString(Path.GetFileNameWithoutExtension(name) + "_" + entryPoint, Path.Combine(_libraryDir, name), source, out var diag);
        if (module is null) throw new KernelCompilationException($"Slang failed to compile {name}:{entryPoint}:\n{diag?.AsString}\nmacros: {FormatMacros(macros)}");
        Check(module.FindEntryPointByName(entryPoint, out var ep), $"find entry point {entryPoint} in {name}", null);
        Check(session.CreateCompositeComponentType([module, ep], out var composite, out diag), "compose", diag);
        Check(composite.Link(out var linked, out diag), "link", diag);
        Check(linked.GetEntryPointCode(0, 0, out var code, out diag), $"codegen {name}:{entryPoint}", diag);
        return code.Buffer.ToArray();

        void Check(SlangResult r, string what, ISlangBlob? d)
        {
            if (r.Failed) throw new KernelCompilationException($"Slang {what} failed ({r}): {d?.AsString}\nmacros: {FormatMacros(macros)}");
        }
    }

    static string FormatMacros(IReadOnlyDictionary<string, string> m) => string.Join(" ", m.Select(kv => $"{kv.Key}={kv.Value}"));

    static SlangCompileTarget Format(ShaderTarget t) => t switch
    {
        ShaderTarget.SpirV => SlangCompileTarget.Spirv,
        ShaderTarget.CudaSource => SlangCompileTarget.CudaSource,
        ShaderTarget.Ptx => SlangCompileTarget.PTX,
        ShaderTarget.Metal => SlangCompileTarget.Metal,
        _ => throw new ArgumentOutOfRangeException(nameof(t)),
    };

    static string Profile(ShaderTarget t) => t switch
    {
        ShaderTarget.SpirV => "spirv_1_6",
        ShaderTarget.CudaSource or ShaderTarget.Ptx => "sm_8_0",
        ShaderTarget.Metal => "metal",
        _ => "",
    };

    static string Extension(ShaderTarget t) => t switch
    {
        ShaderTarget.SpirV => ".spv",
        ShaderTarget.CudaSource => ".cu",
        ShaderTarget.Ptx => ".ptx",
        ShaderTarget.Metal => ".metal",
        _ => ".bin",
    };

    string CacheKey(string source, string entry, ShaderTarget target, IReadOnlyDictionary<string, string> macros)
    {
        var sb = new StringBuilder();
        sb.Append(Version).Append('\n').Append(KernelLibrary.Hash).Append('\n').Append(target).Append('\n').Append(entry).Append('\n');
        foreach (var kv in macros.OrderBy(k => k.Key, StringComparer.Ordinal)) sb.Append(kv.Key).Append('=').Append(kv.Value).Append(';');
        sb.Append('\n').Append(source);
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString())))[..32].ToLowerInvariant();
    }

    public void Dispose() { }
}

public sealed class KernelCompilationException(string message) : Exception(message);

/// <summary>The embedded Slang kernel sources, extracted to a content-addressed directory on first use.</summary>
public static class KernelLibrary
{
    const string Prefix = "Sharplight.Kernels.";
    static readonly Lazy<(string Dir, string Hash)> s_extracted = new(DoExtract);

    public static string Hash => s_extracted.Value.Hash;
    public static string Extract() => s_extracted.Value.Dir;

    public static IEnumerable<string> Files =>
        typeof(KernelLibrary).Assembly.GetManifestResourceNames().Where(n => n.StartsWith(Prefix, StringComparison.Ordinal)).Select(n => n[Prefix.Length..]);

    static (string, string) DoExtract()
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
            foreach (var (k, v) in files) File.WriteAllBytes(Path.Combine(tmp, k), v);
            try { Directory.Move(tmp, dir); }
            catch (IOException) { Directory.Delete(tmp, true); } // another process won the race
        }
        return (dir, hash);
    }
}
