using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Llm.Scoring;

/// <summary>
/// Runs a model's <see cref="WellKnown.ScoreSpansGraph"/> on one prompt per forward pass: the prompt is a single packed sequence, its question
/// and option spans fill the span tables, and the scores of the options come back in option order. A graph with states (a recurrent or hybrid
/// trunk) is run from reset states at offset 0. Not thread-safe; callers serialize.
/// </summary>
internal sealed class SpanScoringRunner : IDisposable
{
    private readonly CompiledModel _compiled;
    private readonly Value _tokens, _positions, _questions, _options;
    private readonly Value? _rowSpans, _seqSpans;
    private readonly int _positionAxes;
    private readonly bool _stateful;
    private readonly RunParameters _run = new();

    /// <summary>Most tokens per prompt.</summary>
    public int MaxTokens { get; }
    /// <summary>Most questions per prompt.</summary>
    public int MaxQuestions { get; }
    /// <summary>Most options per prompt.</summary>
    public int MaxOptions { get; }

    private SpanScoringRunner(CompiledModel compiled, int maxTokens, int maxQuestions, int maxOptions)
    {
        _compiled = compiled;
        var g = compiled.Graph;
        _tokens = g.Input(InputRole.TokenIds);
        _positions = g.Input(InputRole.Positions);
        _rowSpans = g.Inputs.FirstOrDefault(v => v.Input!.Role == InputRole.RowSpans);
        _seqSpans = g.Inputs.FirstOrDefault(v => v.Input!.Role == InputRole.SequenceSpans);
        _positionAxes = (int)_positions.Shape[-1].Fixed;
        _stateful = g.States.Any();
        _questions = g.Input(InputRole.QuestionSpans);
        _options = g.Input(InputRole.OptionSpans);
        MaxTokens = maxTokens;
        MaxQuestions = maxQuestions;
        MaxOptions = maxOptions;
    }

    /// <summary>Compiles the span scoring graph of <paramref name="def"/>.</summary>
    public static SpanScoringRunner Compile(ModelDefinition def, SpanScorerLoadOptions o, Func<IComputeDevice, ICompilationCache?> cacheFor)
    {
        var sw = Stopwatch.StartNew();
        var compiled = CompiledModel.Compile(def.Graph(WellKnown.ScoreSpansGraph), new PlacementOptions
        {
            Devices = [.. o.Devices],
            Fallback = o.Fallback,
            Compile = new CompileOptions
            {
                MaxTokens = o.MaxBatchTokens,
                ContextLength = def.Graph(WellKnown.ScoreSpansGraph).States.Any() ? o.MaxBatchTokens : 1,
                SymbolBounds = new Dictionary<string, long>
                {
                    [WellKnown.SequenceSymbol] = 1, [WellKnown.QuestionSymbol] = o.MaxQuestions, [WellKnown.OptionSymbol] = o.MaxOptions,
                },
                DisabledFeatures = o.DisabledFeatures.ToHashSet(StringComparer.Ordinal),
                CacheFor = cacheFor,
            },
        });
        o.Log?.Invoke($"Compiled '{WellKnown.ScoreSpansGraph}' for {o.MaxBatchTokens} tokens, {o.MaxQuestions} questions, {o.MaxOptions} options in {sw.ElapsedMilliseconds} ms\n{compiled.Plan.Describe()}");
        return new SpanScoringRunner(compiled, o.MaxBatchTokens, o.MaxQuestions, o.MaxOptions);
    }

    /// <summary>The score of every option of <paramref name="prompt"/> (validated by the caller).</summary>
    /// <exception cref="ArgumentException">The prompt exceeds the compiled limits.</exception>
    public float[] Run(SpanPrompt prompt)
    {
        int n = prompt.Tokens.Length;
        if (n > MaxTokens || prompt.Questions.Count > MaxQuestions || prompt.Options.Count > MaxOptions)
        {
            throw new ArgumentException($"The prompt has {n} tokens, {prompt.Questions.Count} questions and {prompt.Options.Count} options; at most "
                + $"{MaxTokens}, {MaxQuestions} and {MaxOptions} fit.");
        }

        // Text positions; of several rope axes (M-RoPE) the first three carry the position.
        var positions = new int[n * _positionAxes];
        var rows = new int[n * 2];
        for (int t = 0; t < n; t++)
        {
            for (int a = 0; a < _positionAxes; a++)
            {
                positions[t * _positionAxes + a] = a < 3 ? t : 0;
            }

            rows[t * 2 + 1] = n;
        }

        // The options of a question are consecutive (checked by SpanPrompt.Validate).
        var questions = new int[prompt.Questions.Count * 6];
        for (int i = 0; i < prompt.Questions.Count; i++)
        {
            var q = prompt.Questions[i];
            questions[i * 6] = q.Start;
            questions[i * 6 + 1] = q.Length;
            questions[i * 6 + 2] = (int)q.Kind;
            questions[i * 6 + 4] = -1;
        }

        for (int i = 0; i < prompt.Options.Count; i++)
        {
            int q = prompt.Options[i].Question * 6;
            if (questions[q + 4] < 0)
            {
                questions[q + 4] = i;
            }

            questions[q + 5]++;
        }

        var options = new int[prompt.Options.Count * 3];
        for (int i = 0; i < prompt.Options.Count; i++)
        {
            var op = prompt.Options[i];
            options[i * 3] = op.Start;
            options[i * 3 + 1] = op.Length;
            options[i * 3 + 2] = op.Question;
        }

        _compiled.SetInput<int>(_tokens, prompt.Tokens);
        _compiled.SetInput<int>(_positions, positions);
        if (_rowSpans is not null)
        {
            _compiled.SetInput<int>(_rowSpans, rows);
        }

        if (_seqSpans is not null)
        {
            _compiled.SetInput<int>(_seqSpans, [0, n]);
        }

        _compiled.SetInput<int>(_questions, questions);
        _compiled.SetInput<int>(_options, options);
        _run.Symbols[WellKnown.TokenSymbol] = n;
        _run.Symbols[WellKnown.SequenceSymbol] = 1;
        _run.Symbols[WellKnown.QuestionSymbol] = prompt.Questions.Count;
        _run.Symbols[WellKnown.OptionSymbol] = prompt.Options.Count;
        if (_stateful)
        {
            _compiled.ResetState();
            _run.Scalars[WellKnown.PastScalar] = 0;
        }

        _compiled.Run(_run);
        var scores = new float[prompt.Options.Count];
        _compiled.GetOutput<float>(WellKnown.ScoresOutput, scores);
        return scores;
    }

    /// <summary>Releases the compiled graph.</summary>
    public void Dispose() => _compiled.Dispose();
}
