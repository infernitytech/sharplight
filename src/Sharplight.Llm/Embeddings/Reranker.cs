using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// Scores how relevant documents are to a query. Cross-encoders (BERT / XLM-RoBERTa classifiers) read "query, document" as one
/// pair through the model's <see cref="WellKnown.ClassifyGraph"/>; decoder rerankers (Qwen3-Reranker) get a judging prompt and
/// score P(yes) from the last token's "yes" / "no" logits. Documents are batched. Thread-safe (calls are serialized).
/// </summary>
public sealed class Reranker : ITextReranker, IDisposable
{
    private const string DefaultInstruction = "Given a web search query, retrieve relevant passages that answer the query";
    private const string YesNoPrefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
    private const string YesNoSuffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    private readonly PackedBatchRunner _runner;
    private readonly Lock _lock = new();
    private readonly string _activation, _pairStyle, _instruction;
    private readonly int _yes, _no;
    private readonly int[] _prefixTokens = [], _suffixTokens = [];

    /// <summary>The model's definition (graphs, weights, metadata).</summary>
    public ModelDefinition Definition { get; }
    /// <summary>The model's tokenizer.</summary>
    public ITokenizer Tokenizer { get; }
    /// <summary>Model name from the file metadata.</summary>
    public string Name { get; }
    /// <summary>How pairs are scored.</summary>
    public RerankerKind Kind { get; }
    /// <summary>Longest (query, document) input in tokens.</summary>
    public int MaxSequenceLength { get; }
    /// <summary>The compiled packed-sequence graph.</summary>
    public PackedBatchRunner Runner => _runner;

    private Reranker(ModelDefinition def, ITokenizer tok, PackedBatchRunner runner, RerankerKind kind, int maxLength, string? instruction)
    {
        Definition = def;
        Tokenizer = tok;
        _runner = runner;
        Kind = kind;
        MaxSequenceLength = maxLength;
        Name = def.Name;
        var md = def.SourceMetadata;
        _activation = md.TryGetValue(CanonicalKeys.ClassifierActivation, out var a) && a is string s ? s : "sigmoid";
        _pairStyle = md.TryGetValue(CanonicalKeys.TokenizerPairStyle, out var ps) && ps is string p ? p
            : md.TryGetValue(CanonicalKeys.TokenizerModel, out var tm) && tm as string == "wordpiece" ? "bert" : "roberta";
        _instruction = instruction ?? (EmbeddingModel.ParsePrompts(md).TryGetValue("query", out var q) ? q : DefaultInstruction);
        if (kind == RerankerKind.YesNo)
        {
            _yes = tok.TokenId("yes") ?? throw new NotSupportedException("The model's vocabulary has no \"yes\" token.");
            _no = tok.TokenId("no") ?? throw new NotSupportedException("The model's vocabulary has no \"no\" token.");
            _prefixTokens = tok.Encode(YesNoPrefix, addSpecial: false);
            _suffixTokens = tok.Encode(YesNoSuffix, addSpecial: false);
        }
    }

    /// <summary>Loads a model (GGUF file, Hugging Face directory, ...) through the registered importers and compiles its classify graph.</summary>
    public static Reranker Load(string path, RerankerLoadOptions options)
    {
        var (def, _, tok) = PackedModelLoadOptions.Open(path, options, new AdapterOptions
        {
            KvCacheType = options.Adapter.KvCacheType, IncludeAuxiliaryGraphs = true, ContextLength = options.Adapter.ContextLength,
        });
        try
        {
            if (!def.Graphs.ContainsKey(WellKnown.ClassifyGraph))
            {
                throw new NotSupportedException($"Model '{def.Name}' ({def.Architecture}) has no '{WellKnown.ClassifyGraph}' graph (no classifier head) and cannot rerank.");
            }

            var kind = options.Kind;
            if (kind == RerankerKind.Auto)
            {
                kind = def.Property(WellKnown.CausalAttention, false) ? RerankerKind.YesNo : RerankerKind.CrossEncoder;
            }

            var runner = PackedBatchRunner.Compile(def, WellKnown.ClassifyGraph, WellKnown.LogitsOutput, options.Devices, options.Fallback,
                options.MaxBatchTokens, options.MaxBatchSequences, options.DisabledFeatures, null, options.Log);
            return new Reranker(def, tok, runner, kind, options.ResolveMaxLength(def), options.Instruction);
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

    /// <summary>The model input of one (query, document) pair.</summary>
    public PackedSequence Encode(string query, string document, RerankOptions? options = null)
    {
        options ??= new RerankOptions();
        if (Kind == RerankerKind.YesNo)
        {
            var body = Tokenizer.Encode($"<Instruct>: {options.Instruction ?? _instruction}\n<Query>: {query}\n<Document>: {document}", addSpecial: false, parseSpecial: false);
            int room = MaxSequenceLength - _prefixTokens.Length - _suffixTokens.Length;
            if (body.Length > room)
            {
                if (!options.Truncate)
                {
                    throw new ArgumentException($"The pair has {body.Length + _prefixTokens.Length + _suffixTokens.Length} tokens; the model accepts at most {MaxSequenceLength}.");
                }

                body = body[..Math.Max(0, room)];
            }
            return new PackedSequence([.. _prefixTokens, .. body, .. _suffixTokens]);
        }

        var a = Tokenizer.Encode(query, addSpecial: false);
        var b = Tokenizer.Encode(document, addSpecial: false);
        int cls = Tokenizer.BosId ?? throw new NotSupportedException("The tokenizer has no classifier (BOS) token.");
        int sep = Definition.SourceMetadata.TryGetValue(CanonicalKeys.TokenizerSeparatorId, out var sv) ? Convert.ToInt32(sv, System.Globalization.CultureInfo.InvariantCulture)
                : Tokenizer.EosId ?? throw new NotSupportedException("The tokenizer has no separator token.");
        bool bert = _pairStyle == "bert";
        int specials = bert ? 3 : 4;
        // Longest-first truncation (Hugging Face's default for pairs): drop tokens from the end of the longer text.
        int over = a.Length + b.Length + specials - MaxSequenceLength;
        if (over > 0)
        {
            if (!options.Truncate)
            {
                throw new ArgumentException($"The pair has {a.Length + b.Length + specials} tokens; the model accepts at most {MaxSequenceLength}.");
            }

            int la = a.Length, lb = b.Length;
            while (over-- > 0) { if (la > lb)
                {
                    la--;
                }
                else
                {
                    lb--;
                }
            }
            a = a[..Math.Max(0, la)];
            b = b[..Math.Max(0, lb)];
        }
        var tokens = new List<int>(a.Length + b.Length + specials) { cls };
        tokens.AddRange(a);
        tokens.Add(sep);
        if (!bert)
        {
            tokens.Add(sep);
        }

        int second = tokens.Count;
        tokens.AddRange(b);
        tokens.Add(sep);
        var types = new int[tokens.Count];
        if (bert)
        {
            for (int i = second; i < types.Length; i++)
            {
                types[i] = 1;
            }
        }

        return new PackedSequence([.. tokens], types);
    }

    /// <summary>Scores every document against the query (batched); results are in document order.</summary>
    public RerankResult Score(string query, IReadOnlyList<string> documents, RerankOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(query);
        var seqs = documents.Select(d => Encode(query, d ?? throw new ArgumentNullException(nameof(documents)), options)).ToList();
        float[][] rows;
        lock (_lock)
        {
            rows = _runner.Run(seqs);
        }

        var scores = new float[rows.Length];
        var logits = new float[rows.Length];
        for (int i = 0; i < rows.Length; i++)
        {
            if (Kind == RerankerKind.YesNo)
            {
                logits[i] = rows[i][_yes] - rows[i][_no];
                scores[i] = Sigmoid(logits[i]);
            }
            else
            {
                logits[i] = rows[i][0];
                scores[i] = _activation switch
                {
                    "sigmoid" => Sigmoid(rows[i][0]),
                    // Multi-label heads: the last label's probability (binary relevant / not relevant classifiers).
                    "softmax" when rows[i].Length > 1 => Softmax(rows[i])[^1],
                    _ => rows[i][0],
                };
            }
        }
        return new RerankResult(scores, logits, [.. seqs.Select(s => s.Tokens.Length)]);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float[] Softmax(float[] x)
    {
        float m = x.Max();
        var e = x.Select(v => MathF.Exp(v - m)).ToArray();
        float s = e.Sum();
        return [.. e.Select(v => v / s)];
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
