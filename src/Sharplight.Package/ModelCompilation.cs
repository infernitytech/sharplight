using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>
/// The <see cref="CompilationWorkload"/> of a package: compiles every graph the way the loaders compile it, so the recorded kernels are
/// the ones a later load asks for: language models through <see cref="LlmModel"/> (decode, decode_multimodal, vision_encode and, with
/// <see cref="PrecompileOptions.Slots"/> above 1, the serving batch decoder), packed-sequence graphs through <see cref="EmbeddingModel"/>
/// (embed) and <see cref="Reranker"/> (classify), the graphs of every component of a model with a <see cref="ModelDescription"/> (speech,
/// image generation, ...), and other graphs (e.g. ONNX models) as a <see cref="CompiledModel"/> with the given symbol bounds plus the
/// component's default bounds.
/// </summary>
/// <remarks>
/// <c>decode_embeddings</c> (the decoder trunk fed with embeddings instead of token ids) is not compiled: no loader compiles it
/// (multimodal prompts use <c>decode_multimodal</c>), so its kernels would never be asked for.
/// </remarks>
/// <param name="packagePath">The package.</param>
/// <param name="definition">The package's model definition.</param>
/// <param name="options">How to compile.</param>
internal sealed class ModelCompilation(string packagePath, ModelDefinition definition, PrecompileOptions options)
{
    private const string MissingBoundPrefix = "No bound";

    /// <summary>The workload delegate.</summary>
    public CompilationWorkload Workload => Run;

    /// <summary>Compiles the package on <paramref name="device"/>, recording into <paramref name="artifacts"/>.</summary>
    /// <exception cref="MissingSymbolBoundException">A non-LLM graph has a symbolic dimension without a bound.</exception>
    public void Run(IComputeDevice device, CompilationArtifacts artifacts, CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        options.Log?.Invoke($"Compiling {packagePath} for {device.Info}");
        if (definition.Description is { } description && description.Components.Any(c => c.Graphs.Count > 0))
        {
            CompileComponents(device, artifacts, description, cancellationToken);
        }
        else if (definition.Graphs.ContainsKey(WellKnown.DecodeGraph) || definition.Graphs.ContainsKey(WellKnown.EmbedGraph)
            || definition.Graphs.ContainsKey(WellKnown.ClassifyGraph))
        {
            CompileLanguageModel(device, artifacts);
        }
        else
        {
            CompileGraphs(device, artifacts, definition.Graphs, new Dictionary<string, long>(), cancellationToken);
        }

        options.Log?.Invoke($"Compiled in {stopwatch.Elapsed.TotalMilliseconds:F0} ms");
    }

    private void CompileLanguageModel(IComputeDevice device, CompilationArtifacts artifacts)
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
                CompileSpeculative(model);
            }
        }

        if (definition.Graphs.ContainsKey(WellKnown.EmbedGraph))
        {
            Optional(WellKnown.EmbedGraph,
                () => EmbeddingModel.Load(packagePath, new EmbeddingLoadOptions { Devices = devices, Log = options.Log, Artifacts = artifacts }));
        }

        if (definition.Graphs.ContainsKey(WellKnown.ClassifyGraph))
        {
            Optional(WellKnown.ClassifyGraph,
                () => Reranker.Load(packagePath, new RerankerLoadOptions { Devices = devices, Log = options.Log, Artifacts = artifacts }));
        }
    }

    /// <summary>
    /// The decoders speculative decoding compiles, sized as their callers size them: the CLI's one-slot decoder and draft head,
    /// and the server's (logits rows for the slots plus the drafts of up to <see cref="Llm.Serving.EngineOptions.SpeculationMaxBatch"/>
    /// speculating sequences).
    /// </summary>
    private void CompileSpeculative(LlmModel model)
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
    private void Optional(string graph, Func<IDisposable> load)
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

    /// <summary>
    /// Compiles the graphs of every component of a described model (<see cref="ModelDescription"/>), each with the given bounds plus the
    /// component's own default bounds (<see cref="ModelComponent.SymbolBounds"/>), so a multi-component model stores the kernels of all
    /// its parts. The kernels of all components go into the same artifact sets (entries are content addressed).
    /// </summary>
    private void CompileComponents(IComputeDevice device, CompilationArtifacts artifacts, ModelDescription description, CancellationToken cancellationToken)
    {
        foreach (var component in description.Components.Where(c => c.Graphs.Count > 0))
        {
            options.Log?.Invoke($"Component '{component.Role}': {string.Join(", ", component.Graphs)}");
            var graphs = component.Graphs.ToDictionary(name => name, definition.Graph, StringComparer.Ordinal);
            CompileGraphs(device, artifacts, graphs, component.SymbolBounds, cancellationToken);
        }
    }

    private void CompileGraphs(IComputeDevice device, CompilationArtifacts artifacts, IReadOnlyDictionary<string, ComputeGraph> graphs,
        IReadOnlyDictionary<string, long> componentBounds, CancellationToken cancellationToken)
    {
        var bounds = new Dictionary<string, long>(options.SymbolBounds, StringComparer.Ordinal);
        foreach (var (symbol, bound) in componentBounds)
        {
            bounds.TryAdd(symbol, bound);
        }

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
            CancellationToken = cancellationToken,
        };
        var fallback = device.Info.Kind == DeviceKind.Cpu ? null : options.FallbackDevice;
        foreach (var (name, graph) in graphs)
        {
            cancellationToken.ThrowIfCancellationRequested();
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
}
