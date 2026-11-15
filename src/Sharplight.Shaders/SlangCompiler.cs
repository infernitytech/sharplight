using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using SlangShaderSharp;

namespace Sharplight.Shaders;

/// <summary>
/// Thin, thread-safe wrapper over the Slang compiler that compiles kernels of the embedded kernel
/// library (or generated source) for any backend target. Kernels are looked up in the model-level cache (artifacts stored with the
/// model) first, then in the persistent per-user compilation cache (<see cref="Runtime.FileCompilationCache"/>: validated, size capped,
/// shared by processes), keyed by the source text, macros, compiler and target, and only then compiled. Shared by all providers.
/// </summary>
public sealed class SlangCompiler : IDisposable
{
    private static long s_compileTicks;
    private static int s_compiled;
    private static int s_diskHits;

    private readonly string _libraryDir;
    /// <summary>One persistent cache per directory in the process, so its size accounting is shared by every compiler using it.</summary>
    private static readonly ConcurrentDictionary<string, Runtime.FileCompilationCache> s_diskCaches = new(StringComparer.Ordinal);

    private readonly Runtime.FileCompilationCache? _disk;
    private readonly ConcurrentDictionary<string, CompiledKernel> _memCache = new();

    /// <summary>Creates a compiler, extracting the embedded kernel library on first use.</summary>
    /// <param name="cacheDirectory">Directory of the persistent compilation cache, shared by processes (null disables it).</param>
    public SlangCompiler(string? cacheDirectory = null)
    {
        _disk = cacheDirectory is null ? null : s_diskCaches.GetOrAdd(Path.GetFullPath(cacheDirectory), directory => new Runtime.FileCompilationCache(directory));
        _libraryDir = KernelLibrary.Extract();
    }

    /// <summary>The persistent compilation cache, or <see langword="null"/> when disabled.</summary>
    public Runtime.FileCompilationCache? DiskCache => _disk;

    /// <summary>Process-wide totals of the kernels this process compiled with Slang or read from the disk cache (diagnostics).</summary>
    public static CompileStatistics Statistics
        => new(Volatile.Read(ref s_compiled), Volatile.Read(ref s_diskHits), TimeSpan.FromSeconds((double)Volatile.Read(ref s_compileTicks) / Stopwatch.Frequency));

    /// <summary>Slang build tag, part of every cache key (the native library ships with the SlangShaderSharp package).</summary>
    private static string Version { get; } = typeof(Slang).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? typeof(Slang).Assembly.GetName().Version?.ToString() ?? "";

    /// <summary>
    /// Name and version of the compiler (<c>slang-&lt;build tag&gt;</c>), recorded in the descriptors of stored code artifacts.
    /// Kernel cache keys include it, so code from another compiler version never hits.
    /// </summary>
    public static string CompilerId => "slang-" + Version;

    /// <summary>Prefix of the model-level cache keys of compiled kernels (<c>kernel/&lt;content hash&gt;</c>).</summary>
    public const string KernelKeyPrefix = "kernel/";

    /// <summary>The Slang profile code for <paramref name="target"/> is generated with (e.g. <c>spirv_1_6</c>).</summary>
    public static string TargetProfile(ShaderTarget target) => Profile(target);

    /// <summary>
    /// The persistent compilation cache directory: the per-user <c>sharplight/kernels</c> directory, or what
    /// <see cref="Runtime.FileCompilationCache.EnvironmentVariable"/> says (another directory, or <see langword="null"/> for <c>off</c>).
    /// </summary>
    public static string? DefaultCacheDirectory => Runtime.FileCompilationCache.ResolveDirectory(
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData, Environment.SpecialFolderOption.Create), "sharplight", "kernels"));

    /// <summary>The cache directory a device with <paramref name="options"/> uses, or <see langword="null"/> when it uses none.</summary>
    public static string? CacheDirectory(Runtime.DeviceOptions options)
        => !options.UseCompilationCache ? null : options.KernelCacheDirectory ?? DefaultCacheDirectory;

    /// <summary>Compiles an entry point of a library kernel file (e.g. "matmul.slang", "matvec").</summary>
    /// <param name="file">Kernel file name in the embedded library.</param>
    /// <param name="entryPoint">Entry point; also defines <c>KERNEL_&lt;ENTRY&gt;</c> for files with several entry points.</param>
    /// <param name="target">Code generation target.</param>
    /// <param name="macros">Preprocessor macros specializing the kernel.</param>
    /// <param name="cache">Optional model-level cache, see <c>Compile</c>.</param>
    /// <returns>The kernel; identical requests return the same instance. It may still be compiling in the background (see <see cref="CompiledKernel.Code"/>).</returns>
    public CompiledKernel CompileLibrary(string file, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros,
        Runtime.ICompilationCache? cache = null)
    {
        var source = KernelLibrary.ReadSource(file);
        // Files with several entry points guard each one with #ifdef KERNEL_<entry>.
        var m = new Dictionary<string, string>(macros) { ["KERNEL_" + entryPoint.ToUpperInvariant()] = "1" };
        return Compile(file, source, KernelLibrary.FileDigest(file), entryPoint, target, m, cache);
    }

    /// <summary>Compiles generated source (may #include library files).</summary>
    /// <param name="name">File name used in diagnostics and dumps.</param>
    /// <param name="source">Slang source text.</param>
    /// <param name="entryPoint">Entry point to compile.</param>
    /// <param name="target">Code generation target.</param>
    /// <param name="macros">Preprocessor macros.</param>
    /// <param name="cache">Optional model-level cache (saved compiled models): consulted first, and given every
    /// kernel resolved so it can be exported.</param>
    /// <returns>The kernel; identical requests return the same instance. Unless it was found in a cache it compiles on a
    /// background thread (see <see cref="SlangWorkerPool"/>): a rejection by Slang surfaces as a
    /// <see cref="KernelCompilationException"/> from <see cref="CompiledKernel.Code"/>.</returns>
    public CompiledKernel Compile(string name, string source, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros,
        Runtime.ICompilationCache? cache = null)
        => Compile(name, source, KernelLibrary.Digest(source), entryPoint, target, macros, cache);

    private CompiledKernel Compile(string name, string source, string sourceDigest, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros,
        Runtime.ICompilationCache? cache)
    {
        var key = CacheKey(sourceDigest, entryPoint, target, macros);
        string ckey = KernelKeyPrefix + key;
        if (DiagnosticSwitches.DumpKernelsDirectory is { } dumpDirectory)
        {
            Directory.CreateDirectory(dumpDirectory);
            File.WriteAllText(Path.Combine(dumpDirectory, $"{Path.GetFileNameWithoutExtension(name)}_{entryPoint}_{key[..12]}.slang"),
                $"// macros: {FormatMacros(macros)}\n{source}");
        }

        var fresh = new CompiledKernel(name, entryPoint, target, key);
        var kernel = _memCache.GetOrAdd(key, fresh);
        if (kernel == fresh)
        {
            Resolve(fresh, source, ckey, macros, cache);
        }

        if (cache is not null && !cache.TryGet(ckey, out _))
        {
            kernel.WhenReady(code => cache.Put(ckey, code));
        }

        return kernel;
    }

    /// <summary>Fills a new kernel from the model cache or the disk cache, or queues its compilation.</summary>
    private void Resolve(CompiledKernel kernel, string source, string modelCacheKey, IReadOnlyDictionary<string, string> macros, Runtime.ICompilationCache? cache)
    {
        if (cache is not null && cache.TryGet(modelCacheKey, out var stored))
        {
            kernel.CompleteFromCache(stored.ToArray());
            return;
        }

        // The disk cache is shared by processes and validates every entry; a damaged or evicted entry is a miss.
        var diskKey = DiskKey(kernel);
        if (_disk is not null && _disk.TryGet(diskKey, out var cached))
        {
            Interlocked.Increment(ref s_diskHits);
            kernel.CompleteFromCache(cached.ToArray());
            return;
        }

        SlangWorkerPool.Shared.Enqueue(
            session =>
            {
                long start = Stopwatch.GetTimestamp();
                var code = CompileUncached(session, kernel.Name, source, kernel.EntryPoint, kernel.Target, macros);
                Interlocked.Add(ref s_compileTicks, Stopwatch.GetTimestamp() - start);
                Interlocked.Increment(ref s_compiled);
                _disk?.Put(diskKey, code);

                kernel.Complete(code);
            },
            error =>
            {
                // A failed kernel must not poison later requests.
                _memCache.TryRemove(new KeyValuePair<string, CompiledKernel>(kernel.CacheKey, kernel));
                kernel.Fail(error);
            });
    }

    /// <summary>Key of a kernel in the persistent cache (the cache key already covers source, macros, compiler and target).</summary>
    private static string DiskKey(CompiledKernel kernel) => $"slang/{kernel.CacheKey}{Extension(kernel.Target)}";

    private byte[] CompileUncached(IGlobalSession gs, string name, string source, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros)
    {
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

    private static string CacheKey(string sourceDigest, string entry, ShaderTarget target, IReadOnlyDictionary<string, string> macros)
    {
        var sb = new StringBuilder();
        sb.Append(Version).Append('\n').Append(sourceDigest).Append('\n').Append(target).Append('\n').Append(entry).Append('\n');
        foreach (var kv in macros.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            sb.Append(kv.Key).Append('=').Append(kv.Value).Append(';');
        }

        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString())))[..32].ToLowerInvariant();
    }

    /// <summary>Does nothing: compiled kernels are plain data and the Slang session is process-wide.</summary>
    public void Dispose()
    {
    }
}
