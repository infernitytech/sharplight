using System.Diagnostics;
using Sharplight.Llm;
using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>
/// Compiles every graph of a package on one device (or for a device profile, without the device) and appends the artifact sets
/// the compilation recorded. Graphs are compiled the way the loaders compile them, so the stored kernels are the ones a later load
/// asks for: language models through <see cref="LlmModel"/> (decode, decode_multimodal, vision_encode and the serving batch
/// decoder), packed-sequence graphs through <see cref="EmbeddingModel"/> (embed) and <see cref="Reranker"/> (classify), other
/// graphs (ONNX models) as a <see cref="CompiledModel"/> with the given symbol bounds and the CPU as fallback.
/// </summary>
/// <remarks>
/// <c>decode_embeddings</c> (the decoder trunk fed with embeddings instead of token ids) is not compiled: no loader compiles it
/// (multimodal prompts use <c>decode_multimodal</c>), so its kernels would never be asked for.
/// </remarks>
internal sealed class PackagePrecompiler(TextWriter output)
{
    private const string TemporaryCachePrefix = "sharplight-precompile-";
    private const long DefaultOnnxBatch = 1;

    public void Run(PrecompileSettings settings)
    {
        // A fresh kernel cache directory: every kernel is compiled (and recorded) from scratch.
        var temporaryCache = settings.UseUserKernelCache ? null : Directory.CreateTempSubdirectory(TemporaryCachePrefix).FullName;
        try
        {
            using var device = OpenDevice(settings, temporaryCache);
            CheckTiers(settings, device);
            output.WriteLine(settings.Profile is { } profile ? $"Compiling {settings.PackagePath} for {profile} (offline)" : $"Compiling {settings.PackagePath} on {device.Info}");
            var source = new ModelPackageImporter().Open(settings.PackagePath);
            try
            {
                // One recording for every load, seeded from the package's own sets; entries served from them point into the
                // package mapping, so they are stored before it is closed.
                var artifacts = new CompilationArtifacts(source.CompilationCaches);
                var definition = source.Definition!;
                var stopwatch = Stopwatch.StartNew();
                if (definition.Graphs.ContainsKey(WellKnown.DecodeGraph) || definition.Graphs.ContainsKey(WellKnown.EmbedGraph)
                    || definition.Graphs.ContainsKey(WellKnown.ClassifyGraph))
                {
                    CompileLanguageModel(settings, device, artifacts, definition);
                }
                else
                {
                    CompileGraphs(settings, device, artifacts, definition);
                }

                output.WriteLine($"Compiled in {stopwatch.ElapsedMilliseconds} ms; slang {Shaders.SlangCompiler.Statistics}");
                Store(settings.PackagePath, artifacts.Export(settings.Tiers));
            }
            finally
            {
                foreach (var resource in source.OwnedResources)
                {
                    resource.Dispose();
                }
            }
        }
        finally
        {
            if (temporaryCache is not null)
            {
                Directory.Delete(temporaryCache, recursive: true);
            }
        }
    }

    private static IComputeDevice OpenDevice(PrecompileSettings settings, string? kernelCacheDirectory)
    {
        var options = new DeviceOptions { KernelCacheDirectory = kernelCacheDirectory };
        foreach (var feature in settings.DisabledFeatures)
        {
            options.DisabledFeatures.Add(feature);
        }

        return settings.Profile is { } profile ? DeviceOpener.OpenProfile(profile, options) : DeviceOpener.Open(settings.Device!.Value, options);
    }

    private static void CheckTiers(PrecompileSettings settings, IComputeDevice device)
    {
        var produced = device.ArtifactTargets.Select(t => t.Tier).ToHashSet(StringComparer.Ordinal);
        if (produced.Count == 0)
        {
            throw new SampleException($"{device.Info} produces no precompiled artifacts.", ExitCodes.InvalidInput);
        }

        var missing = (settings.Tiers ?? new HashSet<string>()).Where(t => !produced.Contains(t)).ToList();
        if (missing.Count > 0)
        {
            throw new SampleException($"{device.Info} does not produce the {string.Join(", ", missing)} tier (it produces {string.Join(", ", produced)}).", ExitCodes.InvalidInput);
        }
    }

    private void CompileLanguageModel(PrecompileSettings settings, IComputeDevice device, CompilationArtifacts artifacts, ModelDefinition definition)
    {
        IReadOnlyList<DeviceSlot> devices = [new DeviceSlot { Device = device }];
        if (definition.Graphs.ContainsKey(WellKnown.DecodeGraph))
        {
            using var model = LlmModel.Load(settings.PackagePath, new LlmLoadOptions
            {
                Devices = devices,
                ContextLength = settings.ContextLength,
                MaxBatchTokens = settings.BatchTokens,
                Log = output.WriteLine,
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

            if (settings.Slots > 1)
            {
                using var decoder = model.CreateBatchDecoder(settings.Slots, settings.ContextLength, kvCacheTokens: settings.KvCacheTokens);
            }
        }

        if (definition.Graphs.ContainsKey(WellKnown.EmbedGraph))
        {
            Optional(WellKnown.EmbedGraph, () => EmbeddingModel.Load(settings.PackagePath, new EmbeddingLoadOptions { Devices = devices, Log = output.WriteLine, Artifacts = artifacts }));
        }

        if (definition.Graphs.ContainsKey(WellKnown.ClassifyGraph))
        {
            Optional(WellKnown.ClassifyGraph, () => Reranker.Load(settings.PackagePath, new RerankerLoadOptions { Devices = devices, Log = output.WriteLine, Artifacts = artifacts }));
        }
    }

    /// <summary>Compiles a graph whose loader may reject this model (e.g. a reranker without "yes"/"no" tokens): reported, not fatal.</summary>
    private void Optional(string graph, Func<IDisposable> load)
    {
        try
        {
            load().Dispose();
        }
        catch (NotSupportedException error)
        {
            output.WriteLine($"Skipped '{graph}': {error.Message}");
        }
    }

    private void CompileGraphs(PrecompileSettings settings, IComputeDevice device, CompilationArtifacts artifacts, ModelDefinition definition)
    {
        var bounds = new Dictionary<string, long>(settings.SymbolBounds, StringComparer.Ordinal);
        if (definition.Properties.TryGetValue(OnnxGraphAdapter.BatchSymbolProperty, out var batch) && batch is string batchSymbol && !bounds.ContainsKey(batchSymbol))
        {
            bounds[batchSymbol] = DefaultOnnxBatch;
            output.WriteLine($"Batch dimension '{batchSymbol}' bounded to {DefaultOnnxBatch} (use --bound {batchSymbol}=VALUE for the batch the model will run with).");
        }

        var compileOptions = new CompileOptions
        {
            MaxTokens = settings.BatchTokens,
            ContextLength = settings.ContextLength,
            SymbolBounds = bounds,
            CacheFor = artifacts.For,
        };
        using var cpu = device.Info.Kind == DeviceKind.Cpu ? null : DeviceOpener.OpenCpu();
        foreach (var (name, graph) in definition.Graphs)
        {
            var stopwatch = Stopwatch.StartNew();
            using var compiled = Compile(name, graph, device, cpu, compileOptions);
            output.WriteLine($"Compiled '{name}' in {stopwatch.ElapsedMilliseconds} ms ({compiled.Plan.Segments.Count} segment(s))");
        }
    }

    private static CompiledModel Compile(string name, Graph.ComputeGraph graph, IComputeDevice device, IComputeDevice? fallback, CompileOptions options)
    {
        try
        {
            return CompiledModel.Compile(graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                // Data-dependent ops the device cannot run go to the CPU, as when ONNX models run.
                Fallback = fallback is null ? null : new DeviceSlot { Device = fallback },
                Compile = options,
            });
        }
        catch (InvalidOperationException error) when (error.Message.StartsWith("No bound", StringComparison.Ordinal))
        {
            throw new SampleException($"Graph '{name}': {error.Message} Give it with --bound SYMBOL=VALUE.", ExitCodes.InvalidInput);
        }
    }

    private void Store(string path, IReadOnlyList<CompilationCache> recorded)
    {
        // A set served entirely from sets already in the package adds nothing (code entries are content addressed).
        var sets = recorded.Where(s => s.FallbackHits < s.Entries.Count).ToList();
        foreach (var unchanged in recorded.Except(sets))
        {
            output.WriteLine($"Already stored: {ArtifactDescriptor.Format(unchanged.Descriptor)} ({unchanged.Entries.Count} entries)");
        }

        if (sets.Count == 0)
        {
            output.WriteLine("Nothing new to store.");
            return;
        }

        var stopwatch = Stopwatch.StartNew();
        ModelPackage.AddArtifacts(path, sets);
        foreach (var set in sets)
        {
            long bytes = set.Entries.Values.Sum(e => (long)e.Length);
            output.WriteLine($"Stored {ArtifactDescriptor.Format(set.Descriptor)}: {set.Entries.Count} entries, {ByteSize.Format(bytes)} "
                + $"({set.FallbackHits} already in the package)");
        }

        output.WriteLine($"{path}: {ByteSize.Format(new FileInfo(path).Length)} [{stopwatch.ElapsedMilliseconds} ms]");
    }
}
