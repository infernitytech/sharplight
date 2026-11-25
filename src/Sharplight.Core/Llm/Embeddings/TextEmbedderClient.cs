using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.Tensors;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// The built-in <see cref="ITextEmbedder"/>: runs the model's <see cref="WellKnown.EmbedGraph"/> (a packed-sequence graph built by the
/// decoder and encoder adapters) over batches of tokenized inputs; outputs are pooled and normalized by the graph as the model declares.
/// </summary>
internal sealed class TextEmbedderClient : ITextEmbedder
{
    /// <summary>Most sequences per forward pass.</summary>
    internal const int MaxBatchSequences = 256;

    private readonly TaskModel _model;
    private readonly PackedBatchRunner _runner;
    private readonly ITokenizer _tokenizer;
    private readonly bool _normalized;
    private readonly string _pooling;
    private readonly string? _defaultPrompt;
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Compiles the embed graph of <paramref name="model"/> onto its devices.</summary>
    /// <exception cref="NotSupportedException">The model has no embed graph.</exception>
    public TextEmbedderClient(TaskModel model)
    {
        _model = model;
        var def = model.Definition;
        if (!def.Graphs.ContainsKey(WellKnown.EmbedGraph))
        {
            throw new NotSupportedException($"Model '{def.Name}' ({def.Architecture}) has no '{WellKnown.EmbedGraph}' graph and cannot produce embeddings.");
        }

        _tokenizer = model.CreateTokenizer();
        _runner = Compile(model, WellKnown.EmbedGraph, WellKnown.EmbeddingsOutput);
        MaxSequenceLength = PackedModelLoadOptions.ResolveMaxLength(def, 0, model.Options.MaxBatchTokens);
        Dimensions = _runner.OutputWidth;
        _pooling = def.Property(WellKnown.EmbeddingPooling, "cls");
        _normalized = def.Property(WellKnown.EmbeddingNormalize, false);
        Prompts = ParsePrompts(def.SourceMetadata);
        _defaultPrompt = def.SourceMetadata.TryGetValue(CanonicalKeys.EmbeddingDefaultPrompt, out var dp) ? dp as string : null;
        Metadata = new TaskClientMetadata(ModelKinds.TextEmbedding, def.Name, def.Architecture);
    }

    public TaskClientMetadata Metadata { get; }

    public int Dimensions { get; }

    public int MaxSequenceLength { get; }

    public IReadOnlyDictionary<string, string> Prompts { get; }

    /// <summary>The compiled packed-sequence graph.</summary>
    internal PackedBatchRunner Runner => _runner;

    /// <summary>Compiles a packed-sequence graph of <paramref name="model"/> for its batch settings (<see cref="TaskLoadOptions.MaxBatchTokens"/> rows).</summary>
    internal static PackedBatchRunner Compile(TaskModel model, string graph, string output)
    {
        int maxTokens = model.Options.MaxBatchTokens;
        var compiled = model.Compile(model.Definition.Graph(graph), new Dictionary<string, long> { [WellKnown.SequenceSymbol] = MaxBatchSequences },
            maxTokens, contextLength: 1);
        try
        {
            return PackedBatchRunner.Create(compiled, output, maxTokens, MaxBatchSequences);
        }
        catch
        {
            compiled.Dispose();
            throw;
        }
    }

    internal static IReadOnlyDictionary<string, string> ParsePrompts(IReadOnlyDictionary<string, object> md)
    {
        var d = new Dictionary<string, string>(StringComparer.Ordinal);
        if (md.TryGetValue(CanonicalKeys.EmbeddingPrompts, out var v) && v is System.Collections.IEnumerable e and not string)
        {
            foreach (var o in e)
            {
                if (o?.ToString() is { } s && s.IndexOf('=') is var i and > 0)
                {
                    d[s[..i]] = s[(i + 1)..];
                }
            }
        }

        return d;
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : serviceKey is null && serviceType.IsInstanceOfType(_tokenizer) ? _tokenizer : null;

    public int[] Tokenize(string text, TextEmbeddingOptions? options = null)
    {
        options ??= new TextEmbeddingOptions();
        string prefix = options.Prefix ?? ((options.Prompt ?? _defaultPrompt) is { } name
            ? Prompts.TryGetValue(name, out var p) ? p : throw new ArgumentException($"The model has no prompt named '{name}' (available: {string.Join(", ", Prompts.Keys)}).")
            : "");
        var body = _tokenizer.Encode(prefix + text, addSpecial: false);
        return WithSpecials(body, options.Truncate);
    }

    private int[] WithSpecials(int[] body, bool truncate)
    {
        bool bos = _tokenizer.AddBos && _tokenizer.BosId is not null;
        bool eos = _tokenizer.AddEos && _tokenizer.EosId is not null;
        int room = MaxSequenceLength - (bos ? 1 : 0) - (eos ? 1 : 0);
        if (body.Length > room)
        {
            if (!truncate)
            {
                throw new ArgumentException($"Input has {body.Length + MaxSequenceLength - room} tokens; the model accepts at most {MaxSequenceLength}.");
            }

            body = body[..Math.Max(0, room)];
        }
        var r = new int[body.Length + (bos ? 1 : 0) + (eos ? 1 : 0)];
        int i = 0;
        if (bos)
        {
            r[i++] = _tokenizer.BosId!.Value;
        }

        body.CopyTo(r, i);
        i += body.Length;
        if (eos)
        {
            r[i] = _tokenizer.EosId!.Value;
        }

        return r;
    }

    public Task<EmbeddingResult> EmbedAsync(IReadOnlyList<string> texts, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        cancellationToken.ThrowIfCancellationRequested();
        options ??= new TextEmbeddingOptions();
        return BackgroundCompute.RunAsync(_gate, () => EmbedSequences([.. texts.Select(t => Tokenize(t, options))], options), cancellationToken);
    }

    public Task<EmbeddingResult> EmbedTokensAsync(IReadOnlyList<IReadOnlyList<int>> tokens, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        cancellationToken.ThrowIfCancellationRequested();
        options ??= new TextEmbeddingOptions();
        var seqs = new int[tokens.Count][];
        for (int i = 0; i < tokens.Count; i++)
        {
            var t = tokens[i].ToArray();
            if (t.Length == 0)
            {
                throw new ArgumentException($"Input {i} is empty.");
            }

            if (t.Length > MaxSequenceLength)
            {
                if (!options.Truncate)
                {
                    throw new ArgumentException($"Input {i} has {t.Length} tokens; the model accepts at most {MaxSequenceLength}.");
                }

                t = t[..MaxSequenceLength];
            }
            foreach (int id in t)
            {
                if ((uint)id >= (uint)_tokenizer.VocabSize)
                {
                    throw new ArgumentException($"Input {i}: token id {id} is outside the vocabulary.");
                }
            }

            seqs[i] = t;
        }
        return BackgroundCompute.RunAsync(_gate, () => EmbedSequences(seqs, options), cancellationToken);
    }

    private EmbeddingResult EmbedSequences(int[][] seqs, TextEmbeddingOptions options)
    {
        if (seqs.Any(s => s.Length == 0))
        {
            throw new ArgumentException("An input has no tokens.");
        }

        var vectors = _runner.Run([.. seqs.Select(s => new PackedSequence(s))]);

        if (options.Dimensions is int dims)
        {
            if (dims <= 0 || dims > Dimensions)
            {
                throw new ArgumentOutOfRangeException(nameof(options), $"dimensions must be between 1 and {Dimensions}.");
            }

            if (_pooling == "none")
            {
                throw new NotSupportedException("Dimension truncation needs pooled embeddings.");
            }

            for (int i = 0; i < vectors.Length; i++)
            {
                var v = vectors[i].AsSpan(0, dims).ToArray();
                if (_normalized)
                {
                    VectorMath.L2Normalize(v);
                }

                vectors[i] = v;
            }
        }
        return new EmbeddingResult(vectors, [.. seqs.Select(s => s.Length)]);
    }

    public void Dispose()
    {
        _runner.Dispose();
        _gate.Dispose();
    }
}
