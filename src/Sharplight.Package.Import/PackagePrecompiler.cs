using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Package.Import;

/// <summary>
/// Compiles every graph of a package on one device (or for a device profile, without the device) and appends the artifact sets
/// the compilation recorded. Graphs are compiled the way the loaders compile them, so the stored kernels are the ones a later load
/// asks for: language models through <see cref="LlmModel"/> (decode, decode_multimodal, vision_encode and, with
/// <see cref="PrecompileOptions.Slots"/> above 1, the serving batch decoder), packed-sequence graphs through
/// <see cref="EmbeddingModel"/> (embed) and <see cref="Reranker"/> (classify), other graphs (e.g. ONNX models) as a
/// <see cref="CompiledModel"/> with the given symbol bounds.
/// </summary>
/// <remarks>
/// <c>decode_embeddings</c> (the decoder trunk fed with embeddings instead of token ids) is not compiled: no loader compiles it
/// (multimodal prompts use <c>decode_multimodal</c>), so its kernels would never be asked for.
/// </remarks>
public static class PackagePrecompiler
{
    private const string TemporaryCachePrefix = "sharplight-precompile-";
    private const string MissingBoundPrefix = "No bound";

    /// <summary>Opens <paramref name="device"/> of <paramref name="provider"/>, compiles the package on it and stores the artifacts.</summary>
    /// <param name="packagePath">The package to compile and append to.</param>
    /// <param name="provider">The execution provider of the device.</param>
    /// <param name="device">The device to compile on.</param>
    /// <param name="options">Compile options, or <see langword="null"/> for the defaults.</param>
    /// <returns>What was stored.</returns>
    /// <exception cref="NotSupportedException">The device produces no precompiled artifacts (e.g. the CPU provider) or not the requested tiers.</exception>
    /// <exception cref="MissingSymbolBoundException">A non-LLM graph has a symbolic dimension without a bound.</exception>
    public static PrecompileResult Precompile(string packagePath, IExecutionProvider provider, DeviceInfo device, PrecompileOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(provider);
        return Precompile(packagePath, deviceOptions => provider.OpenDevice(device, deviceOptions), options);
    }

    /// <summary>
    /// Compiles the package for <paramref name="profile"/> without the device, through the registered provider that compiles for it
    /// (<see cref="ProviderRegistry.OpenProfile"/>), and stores the artifacts (code tier only).
    /// </summary>
    /// <param name="packagePath">The package to compile and append to.</param>
    /// <param name="profile">The device profile to compile for.</param>
    /// <param name="options">Compile options, or <see langword="null"/> for the defaults.</param>
    /// <returns>What was stored.</returns>
    /// <exception cref="NotSupportedException">No registered provider compiles for the profile, or it does not produce the requested tiers.</exception>
    /// <exception cref="MissingSymbolBoundException">A non-LLM graph has a symbolic dimension without a bound.</exception>
    public static PrecompileResult Precompile(string packagePath, DeviceProfile profile, PrecompileOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        return Precompile(packagePath, deviceOptions => ProviderRegistry.OpenProfile(profile, deviceOptions), options);
    }

    /// <summary>Compiles the package on the device <paramref name="openDevice"/> opens and stores the artifacts.</summary>
    /// <param name="packagePath">The package to compile and append to.</param>
    /// <param name="openDevice">Opens the device with the given options (kernel cache directory, disabled features); the device is disposed afterwards.</param>
    /// <param name="options">Compile options, or <see langword="null"/> for the defaults.</param>
    /// <returns>What was stored.</returns>
    /// <exception cref="NotSupportedException">The device produces no precompiled artifacts or not the requested tiers.</exception>
    /// <exception cref="MissingSymbolBoundException">A non-LLM graph has a symbolic dimension without a bound.</exception>
    public static PrecompileResult Precompile(string packagePath, Func<DeviceOptions, IComputeDevice> openDevice, PrecompileOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(openDevice);
        options ??= new PrecompileOptions();

        // A fresh kernel cache directory: every kernel is compiled (and recorded) from scratch.
        var temporaryCache = options.UseUserKernelCache ? null : Directory.CreateTempSubdirectory(TemporaryCachePrefix).FullName;
        try
        {
            var deviceOptions = new DeviceOptions { KernelCacheDirectory = temporaryCache };
            foreach (var feature in options.DisabledFeatures)
            {
                deviceOptions.DisabledFeatures.Add(feature);
            }

            using var device = openDevice(deviceOptions);
            CheckTiers(device, options);
            options.Log?.Invoke($"Compiling {packagePath} for {device.Info}");
            return Compile(packagePath, device, options);
        }
        finally
        {
            if (temporaryCache is not null)
            {
                Directory.Delete(temporaryCache, recursive: true);
            }
        }
    }

    private static void CheckTiers(IComputeDevice device, PrecompileOptions options)
    {
        var produced = device.ArtifactTargets.Select(target => target.Tier).ToHashSet(StringComparer.Ordinal);
        if (produced.Count == 0)
        {
            throw new NotSupportedException($"{device.Info} produces no precompiled artifacts.");
        }

        var missing = (options.Tiers ?? new HashSet<string>()).Where(tier => !produced.Contains(tier)).ToList();
        if (missing.Count > 0)
        {
            throw new NotSupportedException($"{device.Info} does not produce the {string.Join(", ", missing)} tier (it produces {string.Join(", ", produced)}).");
        }
    }

    private static PrecompileResult Compile(string packagePath, IComputeDevice device, PrecompileOptions options)
    {
        var source = new ModelPackageImporter().Open(packagePath);
        try
        {
            // One recording for every load, seeded from the package's own sets; entries served from them point into the package
            // mapping, so they are stored before it is closed.
            var artifacts = new CompilationArtifacts(source.CompilationCaches);
            var definition = source.Definition!;
            var stopwatch = Stopwatch.StartNew();
            if (definition.Graphs.ContainsKey(WellKnown.DecodeGraph) || definition.Graphs.ContainsKey(WellKnown.EmbedGraph)
                || definition.Graphs.ContainsKey(WellKnown.ClassifyGraph))
            {
                CompileLanguageModel(packagePath, device, artifacts, definition, options);
            }
            else
            {
                CompileGraphs(device, artifacts, definition, options);
            }

            var compileTime = stopwatch.Elapsed;
            options.Log?.Invoke($"Compiled in {compileTime.TotalMilliseconds:F0} ms");
            return Store(packagePath, device, artifacts.Export(options.Tiers), compileTime, options);
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    private static void CompileLanguageModel(string packagePath, IComputeDevice device, CompilationArtifacts artifacts, ModelDefinition definition,
        PrecompileOptions options)
    {
        IReadOnlyList<DeviceSlot> devices = [new DeviceSlot { Device = device }];
        if (definition.Graphs.ContainsKey(WellKnown.DecodeGraph))
        {
            using var model = LlmModel.Load(packagePath, new LlmLoadOptions
            {
                Devices = devices,
                ContextLength = options.ContextLength,
                MaxBatchTokens = options.BatchTokens,
                Log = options.Log,
                Artifacts = artifacts,
            });
            foreach (var graph in new[] { WellKnown.DecodeGraph, WellKnown.DecodeMultimodalGraph }.Where(definition.Graphs.ContainsKey))
            {
                using var session = model.CreateSession(graph);
            }

            if (model.SupportsImages)
            {
                using var encoder = model.CreateVisionEncoder();
            }

            if (options.Slots > 1)
            {
                using var decoder = model.CreateBatchDecoder(options.Slots, options.ContextLength, kvCacheTokens: options.KvCacheTokens);
            }

            if (options.DraftTokens > 0)
            {
                CompileSpeculative(model, options);
            }
        }

        if (definition.Graphs.ContainsKey(WellKnown.EmbedGraph))
        {
            Optional(WellKnown.EmbedGraph, options,
                () => EmbeddingModel.Load(packagePath, new EmbeddingLoadOptions { Devices = devices, Log = options.Log, Artifacts = artifacts }));
        }

        if (definition.Graphs.ContainsKey(WellKnown.ClassifyGraph))
        {
            Optional(WellKnown.ClassifyGraph, options,
                () => Reranker.Load(packagePath, new RerankerLoadOptions { Devices = devices, Log = options.Log, Artifacts = artifacts }));
        }
    }

    /// <summary>
    /// The decoders speculative decoding compiles, sized as their callers size them: the CLI's one-slot decoder and draft head,
    /// and the server's (logits rows for the slots plus the drafts of up to <see cref="Llm.Serving.EngineOptions.SpeculationMaxBatch"/>
    /// speculating sequences).
    /// </summary>
    private static void CompileSpeculative(LlmModel model, PrecompileOptions options)
    {
        int k = options.DraftTokens;
        using (model.CreateBatchDecoder(1, options.ContextLength, maxLogitsRows: k + 1, draftTokens: k))
        {
        }

        if (model.HasDraftHead)
        {
            using var head = model.CreateMtpSpeculator(1, k, options.ContextLength);
        }

        if (options.Slots > 1)
        {
            int speculating = Math.Min(options.Slots, new Llm.Serving.EngineOptions().SpeculationMaxBatch);
            using (model.CreateBatchDecoder(options.Slots, options.ContextLength, maxLogitsRows: options.Slots + (speculating * k),
                kvCacheTokens: options.KvCacheTokens, draftTokens: k))
            {
            }

            if (model.HasDraftHead)
            {
                using var head = model.CreateMtpSpeculator(options.Slots, k, options.ContextLength, options.KvCacheTokens);
            }
        }
    }

    /// <summary>Compiles a graph whose loader may reject this model (e.g. a reranker without "yes"/"no" tokens): reported, not fatal.</summary>
    private static void Optional(string graph, PrecompileOptions options, Func<IDisposable> load)
    {
        try
        {
            load().Dispose();
        }
        catch (NotSupportedException error)
        {
            options.Log?.Invoke($"Skipped '{graph}': {error.Message}");
        }
    }

    private static void CompileGraphs(IComputeDevice device, CompilationArtifacts artifacts, ModelDefinition definition, PrecompileOptions options)
    {
        var bounds = new Dictionary<string, long>(options.SymbolBounds, StringComparer.Ordinal);
        foreach (var (symbol, bound) in options.DefaultSymbolBounds?.Invoke(definition) ?? new Dictionary<string, long>())
        {
            if (bounds.TryAdd(symbol, bound))
            {
                options.Log?.Invoke($"Dimension '{symbol}' bounded to {bound} (give a bound for the value the model will run with).");
            }
        }

        var compileOptions = new CompileOptions
        {
            MaxTokens = options.BatchTokens,
            ContextLength = options.ContextLength,
            SymbolBounds = bounds,
            CacheFor = artifacts.For,
            Progress = options.Progress,
            CancellationToken = options.CancellationToken,
        };
        var fallback = device.Info.Kind == DeviceKind.Cpu ? null : options.FallbackDevice;
        foreach (var (name, graph) in definition.Graphs)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            var stopwatch = Stopwatch.StartNew();
            using var compiled = Compile(name, graph, device, fallback, compileOptions);
            options.Log?.Invoke($"Compiled '{name}' in {stopwatch.ElapsedMilliseconds} ms ({compiled.Plan.Segments.Count} segment(s))");
        }
    }

    private static CompiledModel Compile(string name, ComputeGraph graph, IComputeDevice device, IComputeDevice? fallback, CompileOptions options)
    {
        try
        {
            return CompiledModel.Compile(graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                // Data-dependent ops the device cannot run go to the fallback (the CPU), as when such models run.
                Fallback = fallback is null ? null : new DeviceSlot { Device = fallback },
                Compile = options,
            });
        }
        catch (InvalidOperationException error) when (error.Message.StartsWith(MissingBoundPrefix, StringComparison.Ordinal))
        {
            throw new MissingSymbolBoundException(name, error);
        }
    }

    /// <summary>A set served entirely from sets already in the package adds nothing (code entries are content addressed).</summary>
    private static bool HasNewEntries(CompilationCache set) => set.FallbackHits < set.Entries.Count;

    private static PrecompileResult Store(string path, IComputeDevice device, IReadOnlyList<CompilationCache> recorded, TimeSpan compileTime,
        PrecompileOptions options)
    {
        var sets = recorded.Where(HasNewEntries).ToList();
        var unchanged = recorded.Except(sets).Select(set => ArtifactDescriptor.Format(set.Descriptor)).ToList();
        foreach (var set in recorded.Except(sets))
        {
            options.Log?.Invoke($"Already stored: {ArtifactDescriptor.Format(set.Descriptor)} ({set.Entries.Count} entries)");
        }

        if (sets.Count > 0)
        {
            ModelPackage.AddArtifacts(path, sets);
        }

        foreach (var set in sets)
        {
            long bytes = set.Entries.Values.Sum(entry => (long)entry.Length);
            options.Log?.Invoke($"Stored {ArtifactDescriptor.Format(set.Descriptor)}: {set.Entries.Count} entries, {bytes} bytes "
                + $"({set.FallbackHits} already in the package)");
        }

        long storedBytes = sets.Sum(set => set.Entries.Values.Sum(entry => (long)entry.Length));
        var stored = sets.Select(set => ArtifactDescriptor.Format(set.Descriptor)).ToList();
        return new PrecompileResult(device.Info.ToString() ?? "", stored, unchanged, storedBytes, new FileInfo(path).Length, compileTime);
    }
}
