using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// Runs a packed-sequence graph (<see cref="WellKnown.EmbedGraph"/> / <see cref="WellKnown.ClassifyGraph"/>): sequences are
/// concatenated into batches of at most <see cref="MaxBatchTokens"/> rows and <see cref="MaxBatchSequences"/> sequences, each
/// batch is one forward pass, and the per-sequence output rows are returned in input order. Not thread-safe; callers serialize.
/// </summary>
internal sealed class PackedBatchRunner : IDisposable
{
    private readonly CompiledModel _compiled;
    private readonly Value _tokens, _positions, _rowSpans, _seqSpans;
    private readonly Value? _types;
    private readonly string _output;
    private readonly bool _perToken;
    private readonly RunParameters _run = new();
    private int[] _tokBuf, _posBuf, _typeBuf, _rowBuf, _seqBuf;

    /// <summary>Most tokens per forward pass.</summary>
    public int MaxBatchTokens { get; }
    /// <summary>Most sequences per forward pass.</summary>
    public int MaxBatchSequences { get; }
    /// <summary>Width of an output row (hidden size or label count).</summary>
    public int OutputWidth { get; }
    /// <summary>The compiled graph.</summary>
    public CompiledModel Compiled => _compiled;
    /// <summary>Whether the graph emits one row per token (pooling "none") instead of one per sequence.</summary>
    public bool PerToken => _perToken;

    /// <summary>Compiles graph <paramref name="graph"/> of <paramref name="def"/> for batches of up to <paramref name="maxBatchTokens"/> tokens and <paramref name="maxBatchSequences"/> sequences, reading output <paramref name="output"/>.</summary>
    public static PackedBatchRunner Compile(ModelDefinition def, string graph, string output, IReadOnlyList<DeviceSlot> devices, DeviceSlot? fallback,
        int maxBatchTokens, int maxBatchSequences, IReadOnlySet<string>? disabledFeatures, Func<IComputeDevice, ICompilationCache?>? cacheFor, Action<string>? log)
    {
        var sw = Stopwatch.StartNew();
        var compiled = CompiledModel.Compile(def.Graph(graph), new PlacementOptions
        {
            Devices = devices,
            Fallback = fallback,
            Compile = new CompileOptions
            {
                MaxTokens = maxBatchTokens,
                ContextLength = 1,
                SymbolBounds = new Dictionary<string, long> { [WellKnown.SequenceSymbol] = maxBatchSequences },
                DisabledFeatures = disabledFeatures ?? new HashSet<string>(),
                CacheFor = cacheFor,
            },
        });
        log?.Invoke($"Compiled '{graph}' for {maxBatchSequences} sequences / {maxBatchTokens} tokens per batch in {sw.ElapsedMilliseconds} ms\n{compiled.Plan.Describe()}");
        return new PackedBatchRunner(compiled, output, maxBatchTokens, maxBatchSequences);
    }

    private PackedBatchRunner(CompiledModel compiled, string output, int maxTokens, int maxSequences)
    {
        _compiled = compiled;
        var g = compiled.Graph;
        _tokens = g.Input(InputRole.TokenIds);
        _positions = g.Input(InputRole.Positions);
        _rowSpans = g.Input(InputRole.RowSpans);
        _seqSpans = g.Input(InputRole.SequenceSpans);
        _types = g.TryInput(InputRole.TokenTypes);
        _output = output;
        var o = g.Outputs[output];
        _perToken = o.Shape[0].Symbol == WellKnown.TokenSymbol;
        OutputWidth = (int)o.Shape[-1].Fixed;
        MaxBatchTokens = maxTokens;
        MaxBatchSequences = maxSequences;
        _tokBuf = new int[maxTokens];
        _posBuf = new int[maxTokens];
        _typeBuf = new int[maxTokens];
        _rowBuf = new int[maxTokens * 2];
        _seqBuf = new int[maxSequences * 2];
    }

    /// <summary>
    /// Runs all sequences and returns one output per sequence (<see cref="OutputWidth"/> floats, or tokens × width for per-token
    /// graphs). Every sequence must be non-empty and fit <see cref="MaxBatchTokens"/>.
    /// </summary>
    public float[][] Run(IReadOnlyList<PackedSequence> sequences)
    {
        var results = new float[sequences.Count][];
        int i = 0;
        while (i < sequences.Count)
        {
            int start = i, rows = 0;
            while (i < sequences.Count && i - start < MaxBatchSequences)
            {
                int len = sequences[i].Tokens.Length;
                if (len == 0)
                {
                    throw new ArgumentException($"Sequence {i} is empty.");
                }

                if (len > MaxBatchTokens)
                {
                    throw new ArgumentException($"Sequence {i} has {len} tokens; at most {MaxBatchTokens} fit one batch.");
                }

                if (rows + len > MaxBatchTokens)
                {
                    break;
                }

                rows += len;
                i++;
            }
            RunBatch(sequences, start, i - start, rows, results);
        }
        return results;
    }

    private void RunBatch(IReadOnlyList<PackedSequence> seqs, int first, int count, int rows, float[][] results)
    {
        int r = 0;
        for (int s = 0; s < count; s++)
        {
            var seq = seqs[first + s];
            int len = seq.Tokens.Length;
            _seqBuf[s * 2] = r;
            _seqBuf[s * 2 + 1] = len;
            for (int t = 0; t < len; t++)
            {
                _tokBuf[r + t] = seq.Tokens[t];
                _posBuf[r + t] = t;
                _typeBuf[r + t] = seq.TokenTypes is { }
                tt ? tt[t] : 0;
                _rowBuf[(r + t) * 2] = r;
                _rowBuf[(r + t) * 2 + 1] = len;
            }
            r += len;
        }
        _compiled.SetInput<int>(_tokens, _tokBuf.AsSpan(0, rows));
        _compiled.SetInput<int>(_positions, _posBuf.AsSpan(0, rows));
        if (_types is not null)
        {
            _compiled.SetInput<int>(_types, _typeBuf.AsSpan(0, rows));
        }

        _compiled.SetInput<int>(_rowSpans, _rowBuf.AsSpan(0, rows * 2));
        _compiled.SetInput<int>(_seqSpans, _seqBuf.AsSpan(0, count * 2));
        _run.Symbols[WellKnown.TokenSymbol] = rows;
        _run.Symbols[WellKnown.SequenceSymbol] = count;
        _compiled.Run(_run);
        var outRows = new float[(long)(_perToken ? rows : count) * OutputWidth];
        _compiled.GetOutput<float>(_output, outRows);
        for (int s = 0; s < count; s++)
        {
            int row0 = _perToken ? _seqBuf[s * 2] : s, n = _perToken ? _seqBuf[s * 2 + 1] : 1;
            results[first + s] = outRows.AsSpan(row0 * OutputWidth, n * OutputWidth).ToArray();
        }
    }

    /// <summary>Releases the compiled graph.</summary>
    public void Dispose() => _compiled.Dispose();
}
