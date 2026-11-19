using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using SlangShaderSharp;

namespace Sharplight.Shaders;

/// <summary>
/// Thin, thread-safe wrapper over the Slang compiler that compiles kernels of the embedded kernel
/// library (or generated source) for any backend target. Kernels are looked up in the model-level cache first (keyed by the source
/// text, macros, compiler and target: the package's artifact sets, then the per-user <see cref="Runtime.ArtifactStore"/>, see
/// <see cref="Runtime.CompilationArtifacts"/>) and only then compiled. Shared by all providers.
/// </summary>
public sealed class SlangCompiler : IDisposable
{
    private static long s_compileTicks;
    private static int s_compiled;

    private readonly string _libraryDir;
    private readonly ConcurrentDictionary<string, CompiledKernel> _memCache = new();
    private readonly KernelTranslator? _translator;

    /// <summary>Creates a compiler, extracting the embedded kernel library on first use.</summary>
    public SlangCompiler()
        : this(null)
    {
    }

    /// <summary>Creates a compiler whose requests for <see cref="KernelTranslator.Target"/> go through <paramref name="translator"/>.</summary>
    /// <param name="translator">Translates Slang's output before it is cached, or <see langword="null"/>.</param>
    public SlangCompiler(KernelTranslator? translator)
    {
        _libraryDir = KernelLibrary.Extract();
        _translator = translator;
    }

    /// <summary>Process-wide totals of the kernels this process compiled with Slang (diagnostics).</summary>
    public static CompileStatistics Statistics
        => new(Volatile.Read(ref s_compiled), TimeSpan.FromSeconds((double)Volatile.Read(ref s_compileTicks) / Stopwatch.Frequency));

    /// <summary>Slang build tag, part of every cache key (the native library ships with the SlangShaderSharp package).</summary>
    private static string Version { get; } = typeof(Slang).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? typeof(Slang).Assembly.GetName().Version?.ToString() ?? "";

    /// <summary>
    /// Name and version of the compiler (<c>slang-&lt;build tag&gt;</c>), recorded in the descriptors of stored code artifacts.
    /// Kernel cache keys include it, so code from another compiler version never hits.
    /// </summary>
    public static string CompilerId => "slang-" + Version;

    /// <summary>
    /// Start of the first line of every kernel compiled for <see cref="ShaderTarget.Metal"/>, followed by the threadgroup size of the
    /// entry point (<c>// sharplight-threads: 256 1 1</c>): Metal takes it at dispatch, the MSL source does not state it.
    /// </summary>
    public const string MetalThreadsHeader = "// sharplight-threads: ";

    /// <summary>
    /// Start of the first line of every kernel compiled for <see cref="ShaderTarget.CudaSource"/>, followed by the block size of the
    /// entry point (<c>// sharplight-threads: 256 1 1</c>): CUDA takes it at launch, the source does not state it.
    /// </summary>
    public const string CudaThreadsHeader = MetalThreadsHeader;

    /// <summary>Prefix of the model-level cache keys of compiled kernels (<c>kernel/&lt;content hash&gt;</c>).</summary>
    public const string KernelKeyPrefix = "kernel/";

    /// <summary>The Slang profile code for <paramref name="target"/> is generated with (e.g. <c>spirv_1_6</c>).</summary>
    public static string TargetProfile(ShaderTarget target) => Profile(target);

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

    /// <summary>
    /// True when <paramref name="target"/> takes kernels written in CUDA C++ (<see cref="CompileCuda"/>): the compiler translates
    /// CUDA C++ for it (the CUDA provider's NVRTC, see <see cref="KernelTranslator"/>).
    /// </summary>
    public bool CompilesCuda(ShaderTarget target) => _translator is { SourceTarget: ShaderTarget.CudaSource } t && t.Target == target;

    /// <summary>
    /// Compiles a kernel written in CUDA C++ (a <c>.cu</c> file of the kernel library) for a target that takes CUDA C++
    /// (<see cref="CompilesCuda"/>): the file with <paramref name="macros"/> as <c>#define</c> lines and <paramref name="prefix"/>
    /// in front goes to the translator directly, without Slang. Kernels of this kind exist only where Slang cannot express what a
    /// provider offers (asynchronous copies, dynamic shared memory layouts); the lowering selects them through device features.
    /// </summary>
    /// <param name="file">Kernel file name in the embedded library (<c>.cu</c>).</param>
    /// <param name="prefix">Generated source placed after the macros and before the file (e.g. an epilogue function), or empty.</param>
    /// <param name="entryPoint">The <c>extern "C"</c> kernel function.</param>
    /// <param name="threads">Threads per block (x).</param>
    /// <param name="dynamicSharedBytes">Dynamic shared memory the kernel takes at launch (0 for none).</param>
    /// <param name="target">Code generation target.</param>
    /// <param name="macros">Preprocessor macros specializing the kernel.</param>
    /// <param name="cache">Optional model-level cache, consulted first and given the kernel once compiled.</param>
    /// <exception cref="NotSupportedException">The target does not take CUDA C++.</exception>
    public CompiledKernel CompileCuda(string file, string prefix, string entryPoint, int threads, int dynamicSharedBytes, ShaderTarget target,
        IReadOnlyDictionary<string, string> macros, Runtime.ICompilationCache? cache = null)
    {
        if (!CompilesCuda(target))
        {
            throw new NotSupportedException($"Kernel {file} is CUDA C++, which target {target} does not take.");
        }

        var translator = _translator!;
        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $"{CudaThreadsHeader}{threads} 1 1\n");
        if (dynamicSharedBytes > 0)
        {
            text.Append(CultureInfo.InvariantCulture, $"{CudaSharedHeader}{dynamicSharedBytes}\n");
        }

        foreach (var (name, value) in macros.OrderBy(m => m.Key, StringComparer.Ordinal))
        {
            text.Append("#define ").Append(name).Append(' ').Append(value).Append('\n');
        }

        text.Append(prefix).Append('\n').Append(KernelLibrary.ReadSource(file));
        string source = text.ToString();
        string key = CacheKey(KernelLibrary.Digest(source), entryPoint, target, new Dictionary<string, string>(), translator.Id);
        var fresh = new CompiledKernel(file, entryPoint, target, key);
        var kernel = _memCache.GetOrAdd(key, fresh);
        string ckey = KernelKeyPrefix + key;
        if (kernel == fresh)
        {
            if (cache is not null && cache.TryGet(ckey, out var stored))
            {
                fresh.CompleteFromCache(stored.ToArray());
            }
            else
            {
                SlangWorkerPool.Shared.Enqueue(
                    _ => fresh.Complete(translator.Translate(file, entryPoint, Encoding.UTF8.GetBytes(source))),
                    error =>
                    {
                        _memCache.TryRemove(new KeyValuePair<string, CompiledKernel>(key, fresh));
                        fresh.Fail(error);
                    });
            }
        }

        if (cache is not null && !cache.TryGet(ckey, out _))
        {
            kernel.WhenReady(code => cache.Put(ckey, code));
        }

        return kernel;
    }

    /// <summary>
    /// Start of the second header line of a CUDA C++ kernel that takes dynamic shared memory, followed by its bytes
    /// (<c>// sharplight-shared: 101376</c>); the translator keeps it in front of the PTX.
    /// </summary>
    public const string CudaSharedHeader = "// sharplight-shared: ";

    private CompiledKernel Compile(string name, string source, string sourceDigest, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros,
        Runtime.ICompilationCache? cache)
    {
        var translator = _translator is { } t && t.Target == target ? t : null;
        var key = CacheKey(sourceDigest, entryPoint, target, macros, translator?.Id);
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
            Resolve(fresh, source, ckey, macros, cache, translator);
        }

        if (cache is not null && !cache.TryGet(ckey, out _))
        {
            kernel.WhenReady(code => cache.Put(ckey, code));
        }

        return kernel;
    }

    /// <summary>Fills a new kernel from the model-level cache (artifact sets, then the artifact store), or queues its compilation.</summary>
    private void Resolve(CompiledKernel kernel, string source, string modelCacheKey, IReadOnlyDictionary<string, string> macros, Runtime.ICompilationCache? cache,
        KernelTranslator? translator)
    {
        if (cache is not null && cache.TryGet(modelCacheKey, out var stored))
        {
            kernel.CompleteFromCache(stored.ToArray());
            return;
        }

        SlangWorkerPool.Shared.Enqueue(
            session =>
            {
                long start = Stopwatch.GetTimestamp();
                var code = CompileUncached(session, kernel.Name, source, kernel.EntryPoint, translator?.SourceTarget ?? kernel.Target, macros);
                Interlocked.Add(ref s_compileTicks, Stopwatch.GetTimestamp() - start);
                Interlocked.Increment(ref s_compiled);
                if (translator is not null)
                {
                    code = translator.Translate(kernel.Name, kernel.EntryPoint, code);
                }

                kernel.Complete(code);
            },
            error =>
            {
                // A failed kernel must not poison later requests.
                _memCache.TryRemove(new KeyValuePair<string, CompiledKernel>(kernel.CacheKey, kernel));
                kernel.Fail(error);
            });
    }

    private byte[] CompileUncached(IGlobalSession gs, string name, string source, string entryPoint, ShaderTarget target, IReadOnlyDictionary<string, string> macros)
    {
        var desc = new SessionDesc
        {
            Targets = [new TargetDesc { Format = Format(target), Profile = gs.FindProfile(Profile(target)) }],
            SearchPaths = [_libraryDir],
            PreprocessorMacros = [.. macros.Select(kv => new PreprocessorMacroDesc(kv.Key, kv.Value)), .. TargetMacros(target)],
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
        if (target is not ShaderTarget.Metal and not ShaderTarget.CudaSource)
        {
            return code.Buffer.ToArray();
        }

        // MSL and CUDA C++ have no threadgroup-size attribute: the dispatch passes it, so the code carries the entry point's [numthreads]
        // in a header line.
        Span<nuint> threads = stackalloc nuint[3];
        linked.GetLayout(0, out diag).GetEntryPointByIndex(0).GetComputeThreadGroupSize(threads);
        var header = Encoding.UTF8.GetBytes($"{MetalThreadsHeader}{threads[0]} {threads[1]} {threads[2]}\n");
        return [.. header, .. code.Buffer];

        void Check(SlangResult r, string what, ISlangBlob? d)
        {
            if (r.Failed)
            {
                throw new KernelCompilationException($"Slang {what} failed ({r}): {d?.AsString}\nmacros: {FormatMacros(macros)}");
            }
        }
    }

    /// <summary>
    /// Macros naming the code generation target, for the kernels with target-specific forms: <c>SL_TARGET_CUDA</c> for CUDA C++ (the
    /// cooperative-matrix fragments of <c>cuda_mma.slang</c>). The target is part of every cache key, so they are not.
    /// </summary>
    private static IEnumerable<PreprocessorMacroDesc> TargetMacros(ShaderTarget target)
        => target == ShaderTarget.CudaSource ? [new PreprocessorMacroDesc("SL_TARGET_CUDA", "1")] : [];

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

    private static string CacheKey(string sourceDigest, string entry, ShaderTarget target, IReadOnlyDictionary<string, string> macros, string? translatorId)
    {
        var sb = new StringBuilder();
        sb.Append(Version).Append('\n').Append(sourceDigest).Append('\n').Append(target).Append('\n').Append(entry).Append('\n');
        if (translatorId is not null)
        {
            sb.Append(translatorId).Append('\n');
        }

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
