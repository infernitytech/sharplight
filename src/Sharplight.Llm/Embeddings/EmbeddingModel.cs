using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// A text embedding model: encoder embedders (BERT family: bge, e5, MiniLM, nomic-embed, multilingual XLM-R models) and
/// decoder-based ones (Qwen3-Embedding: causal, last-token pooling). Inputs are tokenized, truncated, packed into batches and run
/// through the model's <see cref="WellKnown.EmbedGraph"/>; outputs are pooled and normalized as the model declares (overridable).
/// Thread-safe (calls are serialized).
/// </summary>
public sealed class EmbeddingModel : ITextEmbedder, IDisposable
{
    private readonly PackedBatchRunner _runner;
    private readonly Lock _lock = new();

    /// <summary>The model's definition (graphs, weights, metadata).</summary>
    public ModelDefinition Definition { get; }
    /// <summary>The model's tokenizer.</summary>
    public ITokenizer Tokenizer { get; }
    /// <summary>Model name from the file metadata.</summary>
    public string Name { get; }
    /// <summary>Length of an embedding vector (per token for pooling "none").</summary>
    public int Dimensions { get; }
    /// <summary>How token vectors become one embedding: "cls", "mean", "last" or "none".</summary>
    public string Pooling { get; }
    /// <summary>Whether embeddings are L2-normalized.</summary>
    public bool IsNormalized { get; }
    /// <summary>Longest input in tokens, including special tokens.</summary>
    public int MaxSequenceLength { get; }
    /// <summary>Named prompt prefixes the model was trained with (e.g. "query" → "Instruct: ...\nQuery:").</summary>
    public IReadOnlyDictionary<string, string> Prompts { get; }
    /// <summary>Name of the prompt applied when <see cref="EmbedOptions.Prompt"/> is not set, if the model declares one.</summary>
    public string? DefaultPrompt { get; }
    /// <summary>The compiled packed-sequence graph.</summary>
    internal PackedBatchRunner Runner => _runner;

    private EmbeddingModel(ModelDefinition def, ITokenizer tok, PackedBatchRunner runner, int maxLength)
    {
        Definition = def;
        Tokenizer = tok;
        _runner = runner;
        MaxSequenceLength = maxLength;
        Name = def.Name;
        Dimensions = runner.OutputWidth;
        Pooling = def.Property(WellKnown.EmbeddingPooling, "cls");
        IsNormalized = def.Property(WellKnown.EmbeddingNormalize, false);
        Prompts = ParsePrompts(def.SourceMetadata);
        DefaultPrompt = def.SourceMetadata.TryGetValue(CanonicalKeys.EmbeddingDefaultPrompt, out var dp) ? dp as string : null;
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

    /// <summary>Loads a model (GGUF file, Hugging Face directory, ...) through the registered importers and compiles its embedding graph.</summary>
    public static EmbeddingModel Load(string path, EmbeddingLoadOptions options)
    {
        var adapter = new AdapterOptions
        {
            KvCacheType = options.Adapter.KvCacheType,
            IncludeAuxiliaryGraphs = true,
            ContextLength = options.Adapter.ContextLength,
            EmbeddingPooling = options.Pooling ?? options.Adapter.EmbeddingPooling,
            EmbeddingNormalize = options.Normalize ?? options.Adapter.EmbeddingNormalize,
        };
        var (def, _, tok) = PackedModelLoadOptions.Open(path, options, adapter);
        try
        {
            if (!def.Graphs.ContainsKey(WellKnown.EmbedGraph))
            {
                throw new NotSupportedException($"Model '{def.Name}' ({def.Architecture}) has no '{WellKnown.EmbedGraph}' graph and cannot produce embeddings.");
            }

            var runner = PackedBatchRunner.Compile(def, WellKnown.EmbedGraph, WellKnown.EmbeddingsOutput, options.Devices, options.Fallback,
                options.MaxBatchTokens, options.MaxBatchSequences, options.DisabledFeatures, null, options.Log);
            return new EmbeddingModel(def, tok, runner, options.ResolveMaxLength(def));
        }
        catch
        {
            foreach (var r in def.OwnedResources)
            {
                r.Dispose();
            }

            throw;
        }
    }

    /// <summary>Token ids of one input as the model sees it: prompt prefix, special tokens, truncation.</summary>
    public int[] Tokenize(string text, EmbedOptions? options = null)
    {
        options ??= new EmbedOptions();
        string prefix = options.Prefix ?? ((options.Prompt ?? DefaultPrompt) is { } name
            ? Prompts.TryGetValue(name, out var p) ? p : throw new ArgumentException($"The model has no prompt named '{name}' (available: {string.Join(", ", Prompts.Keys)}).")
            : "");
        var body = Tokenizer.Encode(prefix + text, addSpecial: false);
        return WithSpecials(body, options.Truncate);
    }

    private int[] WithSpecials(int[] body, bool truncate)
    {
        bool bos = Tokenizer.AddBos && Tokenizer.BosId is not null, eos = Tokenizer.AddEos && Tokenizer.EosId is not null;
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
            r[i++] = Tokenizer.BosId!.Value;
        }

        body.CopyTo(r, i);
        i += body.Length;
        if (eos)
        {
            r[i] = Tokenizer.EosId!.Value;
        }

        return r;
    }

    /// <summary>Embeds texts (batched). Vectors are in input order.</summary>
    public EmbeddingResult Embed(IReadOnlyList<string> texts, EmbedOptions? options = null)
    {
        options ??= new EmbedOptions();
        return EmbedSequences([.. texts.Select(t => Tokenize(t, options))], options);
    }

    /// <summary>Embeds one text.</summary>
    public float[] Embed(string text, EmbedOptions? options = null) => Embed([text], options).Vectors[0];

    /// <summary>Embeds pre-tokenized inputs (used as given: no prompt and no special tokens are added; truncated to the model's limit).</summary>
    public EmbeddingResult EmbedTokens(IReadOnlyList<IReadOnlyList<int>> tokens, EmbedOptions? options = null)
    {
        options ??= new EmbedOptions();
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
                if ((uint)id >= (uint)Tokenizer.VocabSize)
                {
                    throw new ArgumentException($"Input {i}: token id {id} is outside the vocabulary.");
                }
            }

            seqs[i] = t;
        }
        return EmbedSequences(seqs, options);
    }

    private EmbeddingResult EmbedSequences(int[][] seqs, EmbedOptions options)
    {
        if (seqs.Any(s => s.Length == 0))
        {
            throw new ArgumentException("An input has no tokens.");
        }

        float[][] vectors;
        lock (_lock)
        {
            vectors = _runner.Run([.. seqs.Select(s => new PackedSequence(s))]);
        }

        if (options.Dimensions is int dims)
        {
            if (dims <= 0 || dims > Dimensions)
            {
                throw new ArgumentOutOfRangeException(nameof(options), $"dimensions must be between 1 and {Dimensions}.");
            }

            if (Pooling == "none")
            {
                throw new NotSupportedException("Dimension truncation needs pooled embeddings.");
            }

            for (int i = 0; i < vectors.Length; i++)
            {
                var v = vectors[i].AsSpan(0, dims).ToArray();
                if (IsNormalized)
                {
                    L2Normalize(v);
                }

                vectors[i] = v;
            }
        }
        return new EmbeddingResult(vectors, [.. seqs.Select(s => s.Length)]);
    }

    internal static void L2Normalize(Span<float> v)
    {
        double s = 0;
        foreach (float x in v)
        {
            s += (double)x * x;
        }

        float inv = s > 0 ? (float)(1 / Math.Sqrt(s)) : 0;
        for (int i = 0; i < v.Length; i++)
        {
            v[i] *= inv;
        }
    }

    /// <summary>Releases the compiled graph and the model files.</summary>
    public void Dispose()
    {
        _runner.Dispose();
        foreach (var r in Definition.OwnedResources)
        {
            r.Dispose();
        }
    }
}
