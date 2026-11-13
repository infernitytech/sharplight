using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using SlangShaderSharp;

namespace Sharplight.Shaders;

/// <summary>
/// Thin, thread-safe wrapper over the Slang compiler that compiles kernels of the embedded kernel
/// library (or generated source) for any backend target, with an on-disk cache keyed by the source
/// text, macros and target. Shared by all providers (Vulkan, CUDA, Metal).
/// </summary>
public sealed class SlangCompiler : IDisposable
{
    private static readonly Lock s_globalLock = new();
    private static IGlobalSession? s_global;

    private readonly string _libraryDir;
    private readonly string? _cacheDir;
    private readonly ConcurrentDictionary<string, CompiledKernel> _memCache = new();
    // Every instance shares the process-wide Slang global session, which is not thread-safe: compile serially
    // across all instances (a per-instance lock let two devices compile concurrently).
    private static readonly Lock _compileLock = new();

    /// <summary>Creates a compiler, extracting the embedded kernel library on first use.</summary>
    /// <param name="cacheDirectory">Directory for compiled binaries, shared by processes (null disables the disk cache).</param>
    public SlangCompiler(string? cacheDirectory = null)
    {
        _cacheDir = cacheDirectory;
        if (_cacheDir is not null)
        {
            Directory.CreateDirectory(_cacheDir);
        }

        _libraryDir = KernelLibrary.Extract();
    }

    private static IGlobalSession Global
    {
        get
        {
            lock (s_globalLock)
            {
                if (s_global is null)
                {
                    var r = Slang.CreateGlobalSession(Slang.ApiVersion, out var gs);
                    if (r.Failed || gs is null)
                    {
                        throw new InvalidOperationException($"Failed to create Slang global session ({r}).");
                    }

                    s_global = gs;
                }
                return s_global;
            }
        }
    }

    private static string Version => Global.GetBuildTagString();

    /// <summary>The per-user kernel cache directory.</summary>
    public static string DefaultCacheDirectory =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData, Environment.SpecialFolderOption.Create), "sharplight", "kernels");

    /// <summary>Compiles an entry point of a library kernel file (e.g. "matmul.slang", "matvec").</summary>
    /// <param name="file">Kernel file name in the embedded library.</param>
    /// <param name="entryPoint">Entry point; also defines <c>KERNEL_&lt;ENTRY&gt;</c> for files with several entry points.</param>
    /// <param name="target">Code generation target.</param>
    /// <param name="macros">Preprocessor macros specializing the kernel.</param>
    /// <param name="cache">Optional model-level cache, see <see cref="Compile"/>.</param>
    /// <returns>The compiled kernel; identical requests return the same instance.</returns>
    /// <exception cref="KernelCompilationException">Slang rejected the kernel.</exception>
    public CompiledKernel CompileLibrary(string file, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros,
        Runtime.ICompilationCache? cache = null)
    {
        var source = File.ReadAllText(Path.Combine(_libraryDir, file));
        // Files with several entry points guard each one with #ifdef KERNEL_<entry>.
        var m = new Dictionary<string, string>(macros) { ["KERNEL_" + entryPoint.ToUpperInvariant()] = "1" };
        return Compile(file, source, entryPoint, target, m, cache);
    }

    /// <summary>Compiles generated source (may #include library files).</summary>
    /// <param name="name">File name used in diagnostics and dumps.</param>
    /// <param name="source">Slang source text.</param>
    /// <param name="entryPoint">Entry point to compile.</param>
    /// <param name="target">Code generation target.</param>
    /// <param name="macros">Preprocessor macros.</param>
    /// <param name="cache">Optional model-level cache (saved compiled models): consulted first, and given every
    /// kernel resolved so it can be exported.</param>
    /// <returns>The compiled kernel; identical requests return the same instance.</returns>
    /// <exception cref="KernelCompilationException">Slang rejected the kernel.</exception>
    public CompiledKernel Compile(string name, string source, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros,
        Runtime.ICompilationCache? cache = null)
    {
        var key = CacheKey(source, entryPoint, target, macros);
        string ckey = "kernel/" + key;
        if (DiagnosticSwitches.DumpKernelsDirectory is { } dumpDirectory)
        {
            Directory.CreateDirectory(dumpDirectory);
            File.WriteAllText(Path.Combine(dumpDirectory, $"{Path.GetFileNameWithoutExtension(name)}_{entryPoint}_{key[..12]}.slang"),
                $"// macros: {FormatMacros(macros)}\n{source}");
        }
        if (_memCache.TryGetValue(key, out var hit))
        {
            if (cache is not null && !cache.TryGet(ckey, out _))
            {
                cache.Put(ckey, hit.Code);
            }

            return hit;
        }
        if (cache is not null && cache.TryGet(ckey, out var stored))
        {
            return _memCache[key] = new CompiledKernel(name, entryPoint, target, stored.ToArray(), key);
        }

        var result = CompileNoModelCache(name, source, entryPoint, target, macros, key);
        cache?.Put(ckey, result.Code);
        return result;
    }

    private CompiledKernel CompileNoModelCache(string name, string source, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros, string key)
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
        {
            code = CompileUncached(name, source, entryPoint, target, macros);
        }

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

    private static byte[]? TryRead(string path)
    {
        try
        {
            return File.Exists(path) ? File.ReadAllBytes(path) : null;
        }
        catch (Exception e) when (e is FileNotFoundException or DirectoryNotFoundException or IOException)
        {
            return null;
        }
    }

    private byte[] CompileUncached(string name, string source, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros)
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
        if (module is null)
        {
            throw new KernelCompilationException($"Slang failed to compile {name}:{entryPoint}:\n{diag?.AsString}\nmacros: {FormatMacros(macros)}");
        }

        Check(module.FindEntryPointByName(entryPoint, out var ep), $"find entry point {entryPoint} in {name}", null);
        Check(session.CreateCompositeComponentType([module, ep], out var composite, out diag), "compose", diag);
        Check(composite.Link(out var linked, out diag), "link", diag);
        Check(linked.GetEntryPointCode(0, 0, out var code, out diag), $"codegen {name}:{entryPoint}", diag);
        return code.Buffer.ToArray();

        void Check(SlangResult r, string what, ISlangBlob? d)
        {
            if (r.Failed)
            {
                throw new KernelCompilationException($"Slang {what} failed ({r}): {d?.AsString}\nmacros: {FormatMacros(macros)}");
            }
        }
    }

    private static string FormatMacros(IReadOnlyDictionary<string, string> m) => string.Join(" ", m.Select(kv => $"{kv.Key}={kv.Value}"));

    private static SlangCompileTarget Format(ShaderTarget t) => t switch
    {
        ShaderTarget.SpirV => SlangCompileTarget.Spirv,
        ShaderTarget.CudaSource => SlangCompileTarget.CudaSource,
        ShaderTarget.Ptx => SlangCompileTarget.PTX,
        ShaderTarget.Metal => SlangCompileTarget.Metal,
        _ => throw new ArgumentOutOfRangeException(nameof(t)),
    };

    private static string Profile(ShaderTarget t) => t switch
    {
        ShaderTarget.SpirV => "spirv_1_6",
        ShaderTarget.CudaSource or ShaderTarget.Ptx => "sm_8_0",
        ShaderTarget.Metal => "metal",
        _ => "",
    };

    private static string Extension(ShaderTarget t) => t switch
    {
        ShaderTarget.SpirV => ".spv",
        ShaderTarget.CudaSource => ".cu",
        ShaderTarget.Ptx => ".ptx",
        ShaderTarget.Metal => ".metal",
        _ => ".bin",
    };

    private string CacheKey(string source, string entry, ShaderTarget target, IReadOnlyDictionary<string, string> macros)
    {
        var sb = new StringBuilder();
        sb.Append(Version).Append('\n').Append(KernelLibrary.Hash).Append('\n').Append(target).Append('\n').Append(entry).Append('\n');
        foreach (var kv in macros.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            sb.Append(kv.Key).Append('=').Append(kv.Value).Append(';');
        }

        sb.Append('\n').Append(source);
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString())))[..32].ToLowerInvariant();
    }

    /// <summary>Does nothing: compiled kernels are plain data and the Slang session is process-wide.</summary>
    public void Dispose()
    {
    }
}
