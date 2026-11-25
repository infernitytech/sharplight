using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// The built-in <see cref="ITextReranker"/>: cross-encoders read "query, document" as one pair through the model's
/// <see cref="WellKnown.ClassifyGraph"/>; decoder rerankers (causal attention) get a judging prompt and score P(yes) from the last
/// token's "yes" / "no" logits. Documents are batched as packed sequences.
/// </summary>
internal sealed class TextRerankerClient : ITextReranker
{
    private const string DefaultInstruction = "Given a web search query, retrieve relevant passages that answer the query";
    private const string YesNoPrefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
    private const string YesNoSuffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    private readonly TaskModel _model;
    private readonly PackedBatchRunner _runner;
    private readonly ITokenizer _tokenizer;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly string _activation, _pairStyle, _instruction;
    private readonly int _yes, _no;
    private readonly int[] _prefixTokens = [];
    private readonly int[] _suffixTokens = [];

    /// <summary>Compiles the classify graph of <paramref name="model"/> onto its devices.</summary>
    /// <exception cref="NotSupportedException">The model has no classify graph, or is a decoder whose vocabulary has no "yes" / "no" tokens.</exception>
    public TextRerankerClient(TaskModel model)
    {
        _model = model;
        var def = model.Definition;
        if (!def.Graphs.ContainsKey(WellKnown.ClassifyGraph))
        {
            throw new NotSupportedException($"Model '{def.Name}' ({def.Architecture}) has no '{WellKnown.ClassifyGraph}' graph (no classifier head) and cannot rerank.");
        }

        _tokenizer = model.CreateTokenizer();
        Kind = def.Property(WellKnown.CausalAttention, false) ? RerankerKind.YesNo : RerankerKind.CrossEncoder;
        var md = def.SourceMetadata;
        _activation = md.TryGetValue(CanonicalKeys.ClassifierActivation, out var a) && a is string s ? s : "sigmoid";
        _pairStyle = md.TryGetValue(CanonicalKeys.TokenizerPairStyle, out var ps) && ps is string p ? p
            : md.TryGetValue(CanonicalKeys.TokenizerModel, out var tm) && tm as string == "wordpiece" ? "bert" : "roberta";
        _instruction = TextEmbedderClient.ParsePrompts(md).TryGetValue("query", out var q) ? q : DefaultInstruction;
        if (Kind == RerankerKind.YesNo)
        {
            _yes = _tokenizer.TokenId("yes") ?? throw new NotSupportedException("The model's vocabulary has no \"yes\" token.");
            _no = _tokenizer.TokenId("no") ?? throw new NotSupportedException("The model's vocabulary has no \"no\" token.");
            _prefixTokens = _tokenizer.Encode(YesNoPrefix, addSpecial: false);
            _suffixTokens = _tokenizer.Encode(YesNoSuffix, addSpecial: false);
        }

        MaxSequenceLength = PackedModelLoadOptions.ResolveMaxLength(def, 0, model.Options.MaxBatchTokens);
        Metadata = new TaskClientMetadata(ModelKinds.TextReranking, def.Name, def.Architecture);
        _runner = TextEmbedderClient.Compile(model, WellKnown.ClassifyGraph, WellKnown.LogitsOutput);
    }

    public TaskClientMetadata Metadata { get; }

    /// <summary>How pairs are scored.</summary>
    public RerankerKind Kind { get; }

    public int MaxSequenceLength { get; }

    /// <summary>The compiled packed-sequence graph.</summary>
    internal PackedBatchRunner Runner => _runner;

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : serviceKey is null && serviceType.IsInstanceOfType(_tokenizer) ? _tokenizer : null;

    /// <summary>The model input of one (query, document) pair.</summary>
    internal PackedSequence Encode(string query, string document, RerankOptions? options = null)
    {
        options ??= new RerankOptions();
        if (Kind == RerankerKind.YesNo)
        {
            var body = _tokenizer.Encode($"<Instruct>: {options.Instruction ?? _instruction}\n<Query>: {query}\n<Document>: {document}", addSpecial: false, parseSpecial: false);
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

        var a = _tokenizer.Encode(query, addSpecial: false);
        var b = _tokenizer.Encode(document, addSpecial: false);
        int cls = _tokenizer.BosId ?? throw new NotSupportedException("The tokenizer has no classifier (BOS) token.");
        int sep = _model.Definition.SourceMetadata.TryGetValue(CanonicalKeys.TokenizerSeparatorId, out var sv) ? Convert.ToInt32(sv, System.Globalization.CultureInfo.InvariantCulture)
                : _tokenizer.EosId ?? throw new NotSupportedException("The tokenizer has no separator token.");
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

            int la = a.Length;
            int lb = b.Length;
            while (over-- > 0)
            {
                if (la > lb)
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

    public Task<RerankResult> RerankAsync(string query, IReadOnlyList<string> documents, RerankOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(documents);
        cancellationToken.ThrowIfCancellationRequested();
        return BackgroundCompute.RunAsync(_gate, () => Score(query, documents, options), cancellationToken);
    }

    private RerankResult Score(string query, IReadOnlyList<string> documents, RerankOptions? options)
    {
        var seqs = documents.Select(d => Encode(query, d ?? throw new ArgumentNullException(nameof(documents)), options)).ToList();
        var rows = _runner.Run(seqs);
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

    public void Dispose()
    {
        _runner.Dispose();
        _gate.Dispose();
    }
}
