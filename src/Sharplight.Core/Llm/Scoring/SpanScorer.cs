using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Scoring;

/// <summary>
/// Runs decision models: a prompt holds questions and their options, and the model's span scoring head gives every option a score (see
/// <see cref="WellKnown.ScoreSpansGraph"/>). Like a reranker it scores rather than generates, but it reads the question and option spans of
/// one prompt instead of (query, document) pairs, so a question's options see each other and the shared context. The prompt is given as
/// <see cref="PromptSegment"/>s (each tokenized on its own, in the model's prompt format) or already tokenized as a <see cref="SpanPrompt"/>.
/// Thread-safe (calls are serialized).
/// </summary>
/// <example>
/// <code>
/// using var scorer = SpanScorer.Load("Decision-2.0-Kai-0.6B.gguf", new SpanScorerLoadOptions { Devices = [new DeviceSlot { Device = gpu }] });
/// var result = scorer.Score([
///     new PromptSegment("Context:\nThe ticket says the app crashes on start.\n\nTask type: choice\nQuestion:\nWhich team?\nOptions:"),
///     new PromptSegment("\n<option>\n"), new PromptSegment("{\"description\":\"Mobile\",\"key\":\"mobile\"}\n</option>", PromptSegmentRole.Option),
///     new PromptSegment("\n<option>\n"), new PromptSegment("{\"description\":\"Billing\",\"key\":\"billing\"}\n</option>", PromptSegmentRole.Option),
///     new PromptSegment("\n\nSelect the single option best supported by the context and instructions.\nDecision:", PromptSegmentRole.Question),
/// ]);
/// int best = result.Best(question: 0);
/// </code>
/// </example>
public sealed class SpanScorer : IDisposable
{
    private readonly SpanScoringRunner _runner;
    private readonly CompilationArtifacts _artifacts;
    private readonly Lock _lock = new();

    /// <summary>The model's definition (graphs, weights, metadata).</summary>
    public ModelDefinition Definition { get; }
    /// <summary>The model's tokenizer.</summary>
    public ITokenizer Tokenizer { get; }
    /// <summary>Model name from the file metadata.</summary>
    public string Name { get; }
    /// <summary>The head type (<see cref="WellKnown.SpanHeadType"/>), e.g. "endpoint".</summary>
    public string HeadType { get; }
    /// <summary>Longest prompt in tokens.</summary>
    public int MaxPromptTokens => _runner.MaxTokens;

    private SpanScorer(ModelDefinition def, ITokenizer tok, SpanScoringRunner runner, CompilationArtifacts artifacts)
    {
        Definition = def;
        Tokenizer = tok;
        Name = def.Name;
        HeadType = def.Property(WellKnown.SpanHeadType, "");
        _runner = runner;
        _artifacts = artifacts;
    }

    /// <summary>Loads a decision model (GGUF file, Hugging Face directory, ...) through the registered importers and compiles its span scoring graph.</summary>
    /// <exception cref="NotSupportedException">The model has no span scoring head.</exception>
    public static SpanScorer Load(string path, SpanScorerLoadOptions options)
    {
        var (def, source, tok) = PackedModelLoader.Open(path, options, new AdapterOptions
        {
            KvCacheType = options.Adapter.KvCacheType, ContextLength = options.Adapter.ContextLength,
        });
        try
        {
            if (!def.Graphs.ContainsKey(WellKnown.ScoreSpansGraph))
            {
                throw new NotSupportedException($"Model '{def.Name}' ({def.Architecture}) has no '{WellKnown.ScoreSpansGraph}' graph (no span scoring head).");
            }

            var artifacts = options.Artifacts ?? new CompilationArtifacts(source.CompilationCaches);
            var runner = SpanScoringRunner.Compile(def, options, artifacts.For);
            return new SpanScorer(def, tok, runner, artifacts);
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

    /// <summary>
    /// Tokenizes <paramref name="segments"/> one by one (BOS first when the tokenizer adds one) and marks the question and option segments as
    /// spans; an option belongs to the last question before it, or to the first question after it when it comes before any question (a prompt
    /// that lists the options before the question, such as Decision 2.0's).
    /// </summary>
    /// <param name="segments">The prompt, segment by segment.</param>
    /// <param name="parseSpecial">
    /// Whether special-token text (e.g. <c>&lt;|im_start|&gt;</c>) in the segments becomes the special token, as Hugging Face tokenizers do by default
    /// and the reference encoders of decision models expect. Pass false for untrusted text.
    /// </param>
    /// <exception cref="ArgumentException">A segment is empty after tokenization, or the spans are invalid.</exception>
    public SpanPrompt Encode(IReadOnlyList<PromptSegment> segments, bool parseSpecial = true)
    {
        var tokens = new List<int>();
        if (Tokenizer.AddBos && Tokenizer.BosId is { } bos)
        {
            tokens.Add(bos);
        }

        var questions = new List<SpanQuestion>();
        var options = new List<SpanOption>();
        var pending = new List<int>();   // options seen before the first question
        foreach (var s in segments)
        {
            var ids = Tokenizer.Encode(s.Text, addSpecial: false, parseSpecial: parseSpecial);
            if (s.Role != PromptSegmentRole.Text && ids.Length == 0)
            {
                throw new ArgumentException($"The {s.Role.ToString().ToLowerInvariant()} segment \"{s.Text}\" has no tokens.");
            }

            if (s.Role == PromptSegmentRole.Question)
            {
                questions.Add(new SpanQuestion(tokens.Count, ids.Length, s.Kind));
                foreach (int i in pending)
                {
                    options[i] = options[i] with { Question = questions.Count - 1 };
                }

                pending.Clear();
            }
            else if (s.Role == PromptSegmentRole.Option)
            {
                if (questions.Count == 0)
                {
                    pending.Add(options.Count);
                }

                options.Add(new SpanOption(tokens.Count, ids.Length, Math.Max(0, questions.Count - 1)));
            }

            tokens.AddRange(ids);
        }

        var prompt = new SpanPrompt([.. tokens], questions, options);
        prompt.Validate();
        return prompt;
    }

    /// <summary>Scores the options of the prompt given as segments (see <see cref="Encode"/>).</summary>
    public SpanScoreResult Score(IReadOnlyList<PromptSegment> segments) => Score(Encode(segments));

    /// <summary>Scores the options of a tokenized prompt; one forward pass over the whole prompt.</summary>
    /// <exception cref="ArgumentException">The spans are invalid or the prompt exceeds the compiled limits.</exception>
    public SpanScoreResult Score(SpanPrompt prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        prompt.Validate();
        float[] scores;
        lock (_lock)
        {
            scores = _runner.Run(prompt);
        }

        return new SpanScoreResult(scores, [.. prompt.Options.Select(o => o.Question)], prompt.Tokens.Length);
    }

    /// <summary>
    /// Compilation artifacts of the compiled graph, one set per device artifact target, seeded from the sets stored with the model. Store them
    /// with the model so later loads skip kernel compilation.
    /// </summary>
    /// <param name="tiers">Only sets of these tiers (<see cref="ArtifactTiers"/>), or <see langword="null"/> for all.</param>
    public IReadOnlyList<CompilationCache> ExportCompilation(IReadOnlySet<string>? tiers = null) => _artifacts.Export(tiers);

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
