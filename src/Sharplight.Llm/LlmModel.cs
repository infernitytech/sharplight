using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>
/// High-level entry point: a loaded decoder-style language model (weights, tokenizer and chat template). Loading reads the
/// model file and builds its compute graphs; nothing is compiled or uploaded to a device until a session, batch decoder or
/// vision encoder is created from it. From a model you create
/// <list type="bullet">
/// <item><description>an <see cref="InferenceSession"/> plus a <see cref="Generator"/> for simple single-sequence generation,</description></item>
/// <item><description>a batch decoder (<see cref="CreateBatchDecoder"/>) for the continuous-batching <see cref="Serving.InferenceEngine"/>,</description></item>
/// <item><description>a <see cref="VisionEncoder"/> (<see cref="CreateVisionEncoder"/>) for image prompts.</description></item>
/// </list>
/// The model owns the weight memory: dispose it after everything created from it has been disposed. Creating sessions and
/// decoders is thread-safe; the created objects are not.
/// </summary>
/// <example>
/// <code>
/// GgufImporter.Register();   // from Sharplight.Gguf; the host registers the formats it supports
/// using var model = LlmModel.Load("qwen3-4b-q4_k_m.gguf", new LlmLoadOptions { Devices = [new DeviceSlot { Device = gpu }] });
/// using var session = model.CreateSession();
/// var generator = new Generator(session, model.Tokenizer);
/// var options = new GenerationOptions { MaxNewTokens = 128, Sampling = SamplingOptions.Greedy };
/// foreach (var token in generator.Generate("The capital of France is", options))
/// {
///     Console.Write(token.Text);
/// }
/// </code>
/// </example>
public sealed class LlmModel : IDisposable
{
    static LlmModel() => BuiltinAdapters.RegisterAll();

    /// <summary>The built model: architecture, hyper-parameters, weights and compute graphs.</summary>
    public ModelDefinition Definition { get; }
    /// <summary>The canonical model source the definition was built from (what a saved package stores).</summary>
    public ModelSource Source { get; }
    /// <summary>The model's tokenizer (shared; safe for concurrent use).</summary>
    public ITokenizer Tokenizer { get; }
    /// <summary>The Jinja chat template stored in the model file (<c>tokenizer.chat_template</c>), or null if it has none.</summary>
    public string? ChatTemplate { get; }
    /// <summary>
    /// Weight buffers shared by everything compiled from this model (sessions, batch decoders, the draft head): on devices that
    /// support it (<see cref="IComputeDevice.SharesWeights"/>) each tensor is uploaded once per device, so e.g. the MTP head reuses
    /// the decoder's embedding and LM head.
    /// </summary>
    public SharedWeights SharedWeights => _sharedWeights;
    private readonly SharedWeights _sharedWeights = new();
    private readonly LlmLoadOptions _options;
    private readonly CompilationArtifacts _artifacts;

    private LlmModel(ModelDefinition def, ModelSource source, ITokenizer tok, LlmLoadOptions options)
    {
        Definition = def;
        Source = source;
        Tokenizer = tok;
        _options = options;
        _artifacts = options.Artifacts ?? new CompilationArtifacts(source.CompilationCaches);
        ChatTemplate = def.SourceMetadata.TryGetValue(CanonicalKeys.ChatTemplate, out var t) ? t as string : null;
    }

    /// <summary>
    /// Loads a model through <see cref="ModelImporterRegistry"/>; the host must have registered an
    /// importer for the file's format (e.g. <c>GgufImporter.Register()</c> from Sharplight.Gguf).
    /// </summary>
    /// <param name="path">Model file (GGUF, ...) or directory (Hugging Face layout), as understood by the registered importers.</param>
    /// <param name="options">Devices and compile settings; also kept for later session / decoder creation.</param>
    /// <exception cref="NotSupportedException">No registered importer or architecture adapter handles the model.</exception>
    public static LlmModel Load(string path, LlmLoadOptions options)
    {
        var sw = Stopwatch.StartNew();
        var ct = options.CancellationToken;
        ct.ThrowIfCancellationRequested();
        var reading = new LoadProgressCounter(options.Progress, LoadStage.Reading, 1, path);
        var sources = ModelImporterRegistry.OpenSet(path, [.. options.Auxiliary]);
        reading.Complete();
        ct.ThrowIfCancellationRequested();
        var building = new LoadProgressCounter(options.Progress, LoadStage.Building, 1);
        var adapter = options.Adapter.With(options.KvCacheType ?? options.Adapter.KvCacheType,
            options.Adapter.ContextLength > 0 ? options.Adapter.ContextLength : options.ContextLength, RecurrentStateType(options));
        var def = ModelAdapterRegistry.Build(sources, adapter, options.Log);
        var kvStates = def.States.Where(st => st.Kind is StateKind.KeyCache or StateKind.ValueCache).ToList();
        int kept = kvStates.Count(st => st.Type != adapter.KvCacheType);
        if (kept > 0 && adapter.KvCacheType.BlockSize > 1)
        {
            options.Log?.Invoke($"KV cache: {kept} of {kvStates.Count} caches stay f16 (rows are not whole {adapter.KvCacheType} blocks).");
        }

        var tok = TokenizerFactory.Create(TokenizerData.FromMetadata(def.SourceMetadata));
        building.Complete();
        options.Log?.Invoke($"Loaded {def.Name} ({def.Architecture}): {def.Weights.Count} tensors, {def.Weights.Values.Sum(w => w.ByteSize) / 1e9:F2} GB, vocab {tok.VocabSize} [{sw.ElapsedMilliseconds} ms]");
        return new LlmModel(def, sources.Primary, tok, options);
    }

    /// <summary>The recurrent state type of <see cref="LlmLoadOptions.RecurrentStateType"/>, after the diagnostic override.</summary>
    private static DataType RecurrentStateType(LlmLoadOptions options)
        => DiagnosticSwitches.RecurrentState ?? options.RecurrentStateType
           ?? (options.AllowReducedPrecision ? DataType.BF16 : options.Adapter.RecurrentStateType);

    /// <summary>
    /// Compiles the decoder onto the configured devices and returns a fresh single-sequence session. The session owns device
    /// memory for its share of the weights, the KV cache and the activations.
    /// </summary>
    /// <param name="graph">The graph to compile: <see cref="WellKnown.DecodeGraph"/> (text) or <see cref="WellKnown.DecodeMultimodalGraph"/> (text and images).</param>
    /// <param name="memoryLimit">Caps the bytes of the primary device used by this session (null: the devices as configured).</param>
    public InferenceSession CreateSession(string graph = WellKnown.DecodeGraph, long? memoryLimit = null)
    {
        var sw = Stopwatch.StartNew();
        var placement = PlacementFor(memoryLimit, _options.ContextLength, rowBound: _options.MaxBatchTokens);
        var compiled = CompiledModel.Compile(WithSamplingHead(Definition.Graph(graph)), placement);
        _options.Log?.Invoke($"Compiled '{graph}' in {sw.ElapsedMilliseconds} ms\n{compiled.Plan.Describe()}");
        CompiledModel? encoder = null;
        if (compiled.Graph.TryInput(InputRole.EncoderOutput) is not null)
        {
            sw.Restart();
            // The source sequence is encoded in one pass (bidirectional attention), up to a context's length.
            int source = MaxSourceTokens(_options.ContextLength);
            encoder = CompiledModel.Compile(Definition.Graph(WellKnown.EncodeGraph), PlacementFor(memoryLimit, _options.ContextLength, 1, source));
            _options.Log?.Invoke($"Compiled '{WellKnown.EncodeGraph}' in {sw.ElapsedMilliseconds} ms\n{encoder.Plan.Describe()}");
        }

        ReportReady(graph);
        return new InferenceSession(compiled, encoder, Definition.Property(WellKnown.DecoderStartToken, 0));
    }

    /// <summary>
    /// True for masked-diffusion language models (Dream, LLaDA; the model has a <see cref="WellKnown.DenoiseGraph"/>): they generate with a
    /// <see cref="DiffusionGenerator"/> (<see cref="CreateDiffusionGenerator"/>) instead of a token-by-token session.
    /// </summary>
    public bool IsDiffusion => Definition.Graphs.ContainsKey(WellKnown.DenoiseGraph);

    /// <summary>
    /// Compiles the denoising pass of a masked-diffusion model onto the configured devices for sequences of up to the context length
    /// (prompt plus generated span).
    /// </summary>
    /// <exception cref="InvalidOperationException">The model is not a diffusion model or its vocabulary has no mask token.</exception>
    public DiffusionGenerator CreateDiffusionGenerator()
    {
        int maskToken = DiffusionMaskToken();
        int context = DiffusionContext(null);
        var runner = PackedBatchRunner.Compile(Definition, WellKnown.DenoiseGraph, WellKnown.LogitsOutput, _options.Devices, _options.Fallback, context, 1,
            _options.DisabledFeatures, CacheFor, _options.Log);
        ReportReady(WellKnown.DenoiseGraph);
        return new DiffusionGenerator(runner, Tokenizer, maskToken, Definition.Property(WellKnown.DiffusionShiftLogits, false));
    }

    /// <summary>
    /// The batch decoder of a masked-diffusion model: the denoising pass compiled for passes of up to <paramref name="context"/> rows holding
    /// up to <paramref name="slots"/> sequences (see <see cref="Serving.DiffusionBatchDecoder"/>).
    /// </summary>
    private Serving.DiffusionBatchDecoder CreateDiffusionDecoder(int slots, int context)
    {
        int maskToken = DiffusionMaskToken();
        var runner = PackedBatchRunner.Compile(Definition, WellKnown.DenoiseGraph, WellKnown.LogitsOutput, _options.Devices, _options.Fallback, context,
            Math.Max(1, slots), _options.DisabledFeatures, CacheFor, _options.Log);
        ReportReady(WellKnown.DenoiseGraph);
        return new Serving.DiffusionBatchDecoder(runner, Math.Max(1, slots), context, maskToken, Definition.Property(WellKnown.DiffusionShiftLogits, false));
    }

    /// <summary>The mask token of a masked-diffusion model.</summary>
    /// <exception cref="InvalidOperationException">The model is not a diffusion model or its vocabulary has no mask token.</exception>
    private int DiffusionMaskToken()
    {
        if (!IsDiffusion)
        {
            throw new InvalidOperationException($"{Definition.Name} is not a masked-diffusion model.");
        }

        if (!Definition.SourceMetadata.TryGetValue(CanonicalKeys.TokenizerMaskId, out var mask))
        {
            throw new InvalidOperationException($"{Definition.Name}: the vocabulary declares no mask token.");
        }

        return Convert.ToInt32(mask, System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>The longest sequence a diffusion pass handles: the requested context, the load options' or the model's.</summary>
    private int DiffusionContext(int? contextLength)
        => contextLength is > 0 ? contextLength.Value : _options.ContextLength > 0 ? _options.ContextLength : Definition.Property(WellKnown.ContextLength, 4096);

    /// <summary>
    /// True for encoder-decoder models (T5, ...; the model has an <see cref="WellKnown.EncodeGraph"/>): a session encodes the prompt as
    /// the source sequence and generates the decoder's output from the start token (see <see cref="InferenceSession.IsEncoderDecoder"/>).
    /// </summary>
    public bool IsEncoderDecoder => Definition.Graphs.ContainsKey(WellKnown.EncodeGraph);

    /// <summary>
    /// Creates the decoder the serving layer batches requests over: with <paramref name="slots"/> &gt; 1 and a graph whose
    /// stateful ops all have multi-sequence forms, a <see cref="Serving.MultiSequenceDecoder"/> (rows of up to
    /// <paramref name="slots"/> sequences per forward pass, each with <paramref name="contextLength"/> tokens of context,
    /// default the load option); otherwise a single-slot <see cref="Serving.SessionBatchDecoder"/>. A masked-diffusion model
    /// (<see cref="IsDiffusion"/>) gets a <see cref="Serving.DiffusionBatchDecoder"/> whose denoising passes hold up to <paramref name="slots"/>
    /// sequences of up to <paramref name="contextLength"/> tokens (prompt plus span; the remaining parameters do not apply).
    /// </summary>
    /// <param name="slots">Sequences decoded together.</param>
    /// <param name="contextLength">Context per sequence (default <see cref="LlmLoadOptions.ContextLength"/>).</param>
    /// <param name="memoryLimit">Device bytes the decoder may use, or null for the devices as configured.</param>
    /// <param name="maxLogitsRows">
    /// Logits rows one step may return (default <paramref name="slots"/>, one per sequence). Speculative decoding needs
    /// <c>slots · (draft tokens + 1)</c>; each extra row costs <c>vocabulary · 4</c> bytes of activations.
    /// </param>
    /// <param name="kvCacheTokens">
    /// Paged KV cache: one pool of this many tokens (in blocks of <see cref="Serving.MultiSequenceDecoder.DefaultKvBlockSize"/>)
    /// shared by all slots, each slot growing up to <paramref name="contextLength"/> as its sequence needs, so many short
    /// sequences fit in the memory of a few long ones. Null: every slot owns <paramref name="contextLength"/> cache rows.
    /// </param>
    /// <param name="draftTokens">
    /// Speculative decoding: the most draft tokens per sequence and step the decoder must verify and roll back (0: none). Models
    /// with recurrent state get their recurrent ops compiled in rollback form for <c>draftTokens + 1</c> rows (see
    /// <see cref="Graph.RecurrentRollback"/>; <see cref="Serving.IBatchDecoder.MaxSpeculativeRows"/>), and a single slot still
    /// gets a multi-sequence decoder. Size <paramref name="maxLogitsRows"/> for the verification rows as well.
    /// </param>
    public Serving.IBatchDecoder CreateBatchDecoder(int slots, int? contextLength = null, long? memoryLimit = null, int? maxLogitsRows = null,
        long? kvCacheTokens = null, int draftTokens = 0)
    {
        if (IsDiffusion)
        {
            return CreateDiffusionDecoder(slots, DiffusionContext(contextLength));
        }

        var graph = Definition.Graph(WellKnown.DecodeGraph);
        string? why = null;
        if (IsEncoderDecoder)
        {
            why = "encoder-decoder model";
        }

        if ((slots <= 1 && draftTokens <= 0) || IsEncoderDecoder || !Graph.MultiSequence.IsSupported(graph, out why))
        {
            if (slots > 1 || draftTokens > 0)
            {
                _options.Log?.Invoke($"Multi-sequence decoding unavailable ({why}); serving one sequence at a time.");
            }

            return new Serving.SessionBatchDecoder(CreateSession(memoryLimit: memoryLimit));
        }
        int ctx = contextLength ?? _options.ContextLength;
        var sw = Stopwatch.StartNew();
        var placement = PlacementFor(memoryLimit, ctx, rowBound: Math.Max(slots, maxLogitsRows ?? 1));
        long paged = kvCacheTokens is > 0 ? PagedRows(kvCacheTokens.Value) : 0;
        var dec = Serving.MultiSequenceDecoder.Create(graph, slots, ctx, _options.MaxBatchTokens, placement, UsesDeviceSampling, paged,
            rollbackRows: draftTokens > 0 ? draftTokens + 1 : 0, hiddenOutput: DraftHiddenSource);
        string kv = paged > 0 ? $", paged KV cache of {paged} tokens" : "";
        _options.Log?.Invoke($"Compiled '{WellKnown.DecodeGraph}' for {slots} sequences x {ctx} tokens{kv} in {sw.ElapsedMilliseconds} ms\n{dec.Compiled.Plan.Describe()}");
        ReportReady(WellKnown.DecodeGraph);
        return dec;
    }

    /// <summary>
    /// True when the model ships a multi-token-prediction draft head (<see cref="WellKnown.DraftGraph"/>), so
    /// <see cref="CreateMtpSpeculator"/> works.
    /// </summary>
    public bool HasDraftHead => Definition.Graphs.ContainsKey(WellKnown.DraftGraph);

    /// <summary>The decode-graph output the draft head reads as hidden states (<see cref="WellKnown.DraftHiddenSource"/>).</summary>
    private string DraftHiddenSource => Definition.Properties.TryGetValue(WellKnown.DraftHiddenSource, out var v) && v is string s ? s : WellKnown.HiddenOutput;

    /// <summary>
    /// Compiles the model's multi-token-prediction head (<see cref="WellKnown.DraftGraph"/>) into a speculator for an
    /// <see cref="Serving.InferenceEngine"/> (<see cref="Serving.EngineOptions.Speculator"/>) over a decoder from
    /// <see cref="CreateBatchDecoder"/> with the same <paramref name="slots"/>, context and <paramref name="draftTokens"/>. The head
    /// has its own KV cache (one attention layer per MTP layer); its embedding and LM head are the decoder's buffers on devices that
    /// share weights (<see cref="SharedWeights"/>).
    /// </summary>
    /// <param name="slots">Slots of the target decoder.</param>
    /// <param name="draftTokens">Drafts proposed per sequence and step (1 to 4 are typical).</param>
    /// <param name="contextLength">Context per sequence (default <see cref="LlmLoadOptions.ContextLength"/>).</param>
    /// <param name="kvCacheTokens">The target decoder's paged KV cache size, or null for a contiguous cache.</param>
    /// <exception cref="NotSupportedException">The model has no draft head.</exception>
    public Serving.Speculative.MtpSpeculator CreateMtpSpeculator(int slots, int draftTokens, int? contextLength = null, long? kvCacheTokens = null)
    {
        if (!HasDraftHead)
        {
            throw new NotSupportedException($"{Definition.Name} has no multi-token-prediction head.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(draftTokens, 1);
        int ctx = contextLength ?? _options.ContextLength;
        var sw = Stopwatch.StartNew();
        var graph = Definition.Graph(WellKnown.DraftGraph);
        var placement = PlacementFor(null, ctx, rowBound: slots);
        // A paged head cache gets a block per slot more than the target's: its rows run up to draftTokens past the target's.
        long paged = kvCacheTokens is > 0 ? PagedRows(kvCacheTokens.Value + (long)slots * (draftTokens + Serving.MultiSequenceDecoder.DefaultKvBlockSize)) : 0;
        // With a device sampling head the draft tokens (greedy) are chosen on the device: only token ids come back.
        var dec = Serving.MultiSequenceDecoder.Create(graph, slots, ctx, _options.MaxBatchTokens, placement, UsesDeviceSampling, paged);
        _options.Log?.Invoke($"Compiled '{WellKnown.DraftGraph}' for {slots} sequences, {draftTokens} drafts, in {sw.ElapsedMilliseconds} ms\n{dec.Compiled.Plan.Describe()}");
        ReportReady(WellKnown.DraftGraph);   // placement, compile and upload progress come through the compile options
        var tokenMap = Definition.Properties.TryGetValue(WellKnown.DraftTokenMap, out var map) ? map as int[] : null;
        return new Serving.Speculative.MtpSpeculator(dec, draftTokens, tokenMap);
    }

    private const long DefaultReserve = 1L << 30;

    /// <summary>
    /// Whether decoders get a device sampling head: enabled by the options and the environment, and the logits are computed on an
    /// accelerator (the last device in pipeline order); on the CPU the head would save nothing over sampling the logits directly.
    /// </summary>
    private bool UsesDeviceSampling => _options.DeviceSampling && DiagnosticSwitches.DeviceSampling
        && _options.Devices.Count > 0 && _options.Devices[^1].Device.Info.Kind != DeviceKind.Cpu;

    private ComputeGraph WithSamplingHead(ComputeGraph graph)
        => UsesDeviceSampling && SamplingHead.CanAppend(graph) ? SamplingHead.Append(graph) : graph;

    /// <summary>Device placement and compile settings shared by sessions and batch decoders.</summary>
    /// <param name="memoryLimit">Bytes the primary device may hold, or null for the devices as configured.</param>
    /// <param name="contextLength">KV-cache rows per sequence.</param>
    /// <param name="rowBound">Upper bound of the <c>R</c> symbol (rows whose logits are read per step).</param>
    /// <param name="maxTokens">Rows per forward pass (default <see cref="LlmLoadOptions.MaxBatchTokens"/>).</param>
    private PlacementOptions PlacementFor(long? memoryLimit, int contextLength, long rowBound, long? maxTokens = null) => new()
    {
        Devices = DevicesWithLimit(memoryLimit),
        Fallback = _options.Fallback,
        ReservePerDevice = _options.ReservePerDevice ?? (memoryLimit is null ? DefaultReserve : 0),
        Compile = new CompileOptions
        {
            MaxTokens = maxTokens ?? _options.MaxBatchTokens,
            ContextLength = contextLength,
            SymbolBounds = new Dictionary<string, long> { ["R"] = rowBound, [WellKnown.EncoderSymbol] = MaxSourceTokens(contextLength) },
            DisabledFeatures = _options.DisabledFeatures,
            AllowReducedPrecision = _options.AllowReducedPrecision,
            CacheFor = CacheFor,
            Progress = _options.Progress,
            CancellationToken = _options.CancellationToken,
            SharedWeights = _sharedWeights,
        },
    };

    /// <summary>Longest source sequence of an encoder-decoder model: a context's length, at least a batch.</summary>
    private int MaxSourceTokens(int contextLength) => Math.Max(_options.MaxBatchTokens, contextLength);

    private void ReportReady(string graph) => new LoadProgressCounter(_options.Progress, LoadStage.Ready, 1, graph).Complete();

    /// <summary>
    /// Caps the first (primary) device's share of the model at <paramref name="memoryLimit"/> bytes (layers beyond it go to
    /// the fallback device or the load fails); other devices keep their own limits. Null = the devices as configured.
    /// </summary>
    private IReadOnlyList<DeviceSlot> DevicesWithLimit(long? memoryLimit)
    {
        if (memoryLimit is not { } limit || _options.Devices.Count == 0)
        {
            return _options.Devices;
        }

        var d0 = _options.Devices[0];
        var first = new DeviceSlot { Device = d0.Device, Weight = d0.Weight, MemoryLimit = d0.MemoryLimit is { } m ? Math.Min(m, limit) : limit };
        return [first, .. _options.Devices.Skip(1)];
    }

    /// <summary>
    /// Device memory needed to serve this model with <paramref name="slots"/> concurrent sequences of
    /// <paramref name="contextLength"/> tokens (default the load option): weights + KV/recurrent state + activation bound.
    /// Computed from the graph alone; nothing is allocated on a device. For a masked-diffusion model: weights + the activations of a
    /// denoising pass over <paramref name="contextLength"/> rows (it has no per-sequence state).
    /// </summary>
    /// <param name="slots">Concurrent sequences.</param>
    /// <param name="contextLength">Context per sequence (default <see cref="LlmLoadOptions.ContextLength"/>).</param>
    /// <param name="kvCacheTokens">Paged KV cache of this many tokens shared by the slots (see <see cref="CreateBatchDecoder"/>), or null.</param>
    /// <param name="draftTokens">
    /// Speculative decoding (as <see cref="CreateBatchDecoder"/>'s <c>draftTokens</c>): adds the rollback record states of the
    /// recurrent ops (<see cref="Graph.RecurrentRollback.RecordStateBytes"/>) and sizes the logits rows for verification.
    /// </param>
    /// <param name="draftHead">Also count the MTP draft head (<see cref="CreateMtpSpeculator"/>): its own weights (the ones it shares
    /// with the decoder are counted once), its KV cache and activations.</param>
    /// <param name="maxLogitsRows">Logits rows per step (default <paramref name="slots"/> · (<paramref name="draftTokens"/> + 1)).</param>
    public ModelFootprint EstimateFootprint(int slots = 1, int? contextLength = null, long? kvCacheTokens = null, int draftTokens = 0, bool draftHead = false,
        int? maxLogitsRows = null)
    {
        if (IsDiffusion)
        {
            // One denoising pass over up to the context's rows; no per-sequence state.
            return PlacementPlanner.Estimate(Definition.Graph(WellKnown.DenoiseGraph), new CompileOptions
            {
                MaxTokens = DiffusionContext(contextLength),
                ContextLength = 1,
                SymbolBounds = new Dictionary<string, long> { [WellKnown.SequenceSymbol] = Math.Max(1, slots) },
            }, 1, 0);
        }

        int ctx = contextLength ?? _options.ContextLength;
        var graph = Definition.Graph(WellKnown.DecodeGraph);
        bool multi = (slots > 1 || draftTokens > 0) && Graph.MultiSequence.IsSupported(graph, out _);
        long paged = multi && kvCacheTokens is > 0 ? PagedRows(kvCacheTokens.Value) : 0;
        long rows = maxLogitsRows ?? (long)slots * (Math.Max(0, draftTokens) + 1);
        var fp = PlacementPlanner.Estimate(graph, new CompileOptions
        {
            MaxTokens = _options.MaxBatchTokens,
            ContextLength = ctx,
            SymbolBounds = new Dictionary<string, long> { ["R"] = multi ? Math.Min(rows, _options.MaxBatchTokens) : _options.MaxBatchTokens },
        }, multi ? slots : 1, paged);
        if (!multi || draftTokens <= 0)
        {
            return fp;
        }

        long records = Graph.RecurrentRollback.RecordStateBytes(graph, slots, draftTokens + 1);
        fp = fp with { StateBytes = fp.StateBytes + records };
        if (!draftHead || !HasDraftHead)
        {
            return fp;
        }

        // The head (see CreateMtpSpeculator): its paged cache runs a block per slot ahead of the decoder's; the embedding and LM
        // head are the decoder's buffers (shared weights), so only its own weights count.
        var head = Definition.Graph(WellKnown.DraftGraph);
        long headPaged = paged > 0 ? PagedRows(kvCacheTokens!.Value + (long)slots * (draftTokens + Serving.MultiSequenceDecoder.DefaultKvBlockSize)) : 0;
        var headFp = PlacementPlanner.Estimate(head, new CompileOptions
        {
            MaxTokens = _options.MaxBatchTokens,
            ContextLength = ctx,
            SymbolBounds = new Dictionary<string, long> { ["R"] = slots },
        }, slots, headPaged);
        var decoderSources = WeightsOf(graph).Select(w => w.Source).ToHashSet(ReferenceEqualityComparer.Instance);
        long shared = WeightsOf(head).Where(w => decoderSources.Contains(w.Source)).DistinctBy(w => w.Source, ReferenceEqualityComparer.Instance).Sum(w => w.ByteSize);
        return new ModelFootprint(fp.WeightBytes + headFp.WeightBytes - shared, fp.StateBytes + headFp.StateBytes, fp.ActivationBytes + headFp.ActivationBytes);
    }

    private static IEnumerable<WeightInfo> WeightsOf(ComputeGraph graph)
        => graph.Nodes.SelectMany(n => n.Inputs).Select(v => v.Weight).OfType<WeightInfo>().Distinct();

    /// <summary>Cache rows of a paged KV cache of <paramref name="tokens"/> tokens (whole blocks).</summary>
    private static long PagedRows(long tokens)
    {
        long b = Serving.MultiSequenceDecoder.DefaultKvBlockSize;
        return (tokens + b - 1) / b * b;
    }

    /// <summary>True when the model has a vision tower (loaded from a file containing one, or together with a "mmproj" auxiliary file).</summary>
    public bool SupportsImages => Definition.Graphs.ContainsKey(WellKnown.VisionEncodeGraph);

    /// <summary>
    /// Compiles the vision encoder onto the configured devices. The returned encoder turns <see cref="ImageInput"/>s into
    /// embeddings; pass it to a <see cref="Generator"/> (with a session created on <see cref="WellKnown.DecodeMultimodalGraph"/>).
    /// </summary>
    public VisionEncoder CreateVisionEncoder()
    {
        var sw = Stopwatch.StartNew();
        var enc = VisionEncoder.Create(Definition, _options.Devices, _options.Fallback, _options.ImageTokens, _options.DisabledFeatures, CacheFor);
        _options.Log?.Invoke($"Compiled '{WellKnown.VisionEncodeGraph}' (up to {enc.MaxPatches / enc.Parameters.MergeUnit} image tokens) in {sw.ElapsedMilliseconds} ms\n{enc.Compiled.Plan.Describe()}");
        return enc;
    }

    /// <summary>The device's recording cache, seeded from the matching artifact sets stored in the model file.</summary>
    private ICompilationCache? CacheFor(IComputeDevice device) => _artifacts.For(device);

    /// <summary>
    /// Compilation artifacts of every graph compiled so far (call after <see cref="CreateSession"/>), one set per device artifact
    /// target (<see cref="IComputeDevice.ArtifactTargets"/>: e.g. a code tier with the kernels and a binary tier with the driver
    /// pipeline cache). Store them with the model source (e.g. Sharplight.Package's <c>ModelPackage.Save</c> or
    /// <c>ModelPackage.AddArtifacts</c>) so later loads skip kernel compilation.
    /// </summary>
    /// <param name="tiers">Only sets of these tiers (<see cref="ArtifactTiers"/>), or <see langword="null"/> for all.</param>
    public IReadOnlyList<CompilationCache> ExportCompilation(IReadOnlySet<string>? tiers = null) => _artifacts.Export(tiers);

    /// <summary>Releases the weight memory (mapped files, host buffers). Sessions and decoders created from the model must be disposed first.</summary>
    public void Dispose()
    {
        foreach (var r in Definition.OwnedResources)
        {
            r.Dispose();
        }
    }
}
