using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Llm.Serving;

/// <summary>
/// <see cref="IBatchDecoder"/> that runs rows of several sequences in one forward pass (continuous batching): the
/// decoder graph rewritten by <see cref="MultiSequence.Transform"/> and compiled with <c>slots · context</c> cache rows.
/// Each step packs every input's rows consecutively and fills the sequence row / run tables, positions and output rows.
/// A slot whose next input starts at position 0 starts from empty state (recurrent state is cleared by the kernels,
/// KV rows past a sequence's position are never read), so <see cref="Reset"/> costs nothing.
/// </summary>
public sealed class MultiSequenceDecoder : IBatchDecoder
{
    private readonly CompiledModel _compiled;
    private readonly Value _tokens, _rows, _runs;
    private readonly Value? _positions, _outputRows;
    private readonly int _posAxes;
    private readonly int[] _length;
    private readonly RunParameters _run = new();
    private readonly int[] _tokBuf, _posBuf, _rowBuf, _outBuf;
    private readonly int[] _runBuf;
    private readonly float[] _logits;
    private readonly List<float[]> _results = [];

    public int SlotCount { get; }
    public int ContextLength { get; }
    public int MaxBatchTokens { get; }
    public int VocabSize { get; }
    public bool BatchesSequences => true;
    public bool CanTruncate { get; }
    public CompiledModel Compiled => _compiled;
    public long DeviceMemoryBytes => _compiled.AcceleratorMemoryBytes;

    /// <param name="compiled">A model compiled from <see cref="MultiSequence.Transform"/>'s graph with context length
    /// <paramref name="slots"/> · <paramref name="contextLength"/> and symbol bounds U = slots, R ≥ slots.</param>
    /// <param name="hasRecurrentState">Whether a slot holds state other than KV rows (then contexts cannot be truncated).</param>
    public MultiSequenceDecoder(CompiledModel compiled, int slots, int contextLength, bool hasRecurrentState)
    {
        _compiled = compiled;
        var g = compiled.Graph;
        _tokens = g.Input(InputRole.TokenIds);
        _rows = g.Input(InputRole.SequenceRows);
        _runs = g.Input(InputRole.SequenceRuns);
        _positions = g.TryInput(InputRole.Positions);
        _outputRows = g.TryInput(InputRole.OutputRows);
        if (g.TryInput(InputRole.Embeddings) is not null)
        {
            throw new NotSupportedException("Multimodal graphs are not supported by the multi-sequence decoder.");
        }

        _posAxes = _positions is null ? 0 : _positions.Rank == 1 ? 1 : (int)_positions.Shape[-1].Fixed;
        SlotCount = slots;
        ContextLength = contextLength;
        MaxBatchTokens = (int)compiled.Options.MaxTokens;
        VocabSize = (int)g.Outputs["logits"].Shape[-1].Fixed;
        CanTruncate = !hasRecurrentState;
        _length = new int[slots];
        _tokBuf = new int[MaxBatchTokens];
        _posBuf = new int[MaxBatchTokens * Math.Max(1, _posAxes)];
        _rowBuf = new int[MaxBatchTokens * 4];
        _runBuf = new int[slots * 4];
        _outBuf = new int[slots];
        _logits = new float[(long)slots * VocabSize > int.MaxValue ? throw new NotSupportedException("Logits buffer too large.") : slots * VocabSize];
        for (int i = 0; i < slots; i++)
        {
            _results.Add(new float[VocabSize]);
        }
    }

    /// <summary>Compiles <paramref name="graph"/> (single-sequence decoder) for <paramref name="slots"/> sequences.</summary>
    public static MultiSequenceDecoder Create(ComputeGraph graph, int slots, int contextLength, int maxBatchTokens, PlacementOptions placement)
    {
        var ms = MultiSequence.Transform(graph, slots);
        var c = placement.Compile;
        var bounds = new Dictionary<string, long>(c.SymbolBounds)
        {
            [WellKnown.RunSymbol] = slots,
            ["R"] = Math.Max(slots, c.SymbolBounds.GetValueOrDefault("R", 1)),
        };
        var compiled = CompiledModel.Compile(ms.Graph, new PlacementOptions
        {
            Devices = placement.Devices,
            Fallback = placement.Fallback,
            ReservePerDevice = placement.ReservePerDevice,
            Compile = new CompileOptions
            {
                MaxTokens = maxBatchTokens,
                ContextLength = (long)slots * contextLength,
                SymbolBounds = bounds,
                DisabledFeatures = c.DisabledFeatures,
                AllowReducedPrecision = c.AllowReducedPrecision,
                CacheFor = c.CacheFor,
            },
        });
        bool recurrent = ms.States.Keys.Any(s => !MultiSequence.IsKvState(s));
        return new MultiSequenceDecoder(compiled, slots, contextLength, recurrent);
    }

    public int Length(int slot) { Check(slot); return _length[slot]; }

    public void Reset(int slot) { Check(slot); _length[slot] = 0; }

    public void Truncate(int slot, int length)
    {
        Check(slot);
        if (length < 0 || length > _length[slot])
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        if (length != _length[slot] && length != 0 && !CanTruncate)
        {
            throw new NotSupportedException("This model's recurrent state cannot be truncated.");
        }

        _length[slot] = length;
    }

    public IReadOnlyList<float[]?> Step(IReadOnlyList<SequenceInput> inputs)
    {
        var res = new float[]?[inputs.Count];
        if (inputs.Count == 0)
        {
            return res;
        }

        if (inputs.Count > SlotCount)
        {
            throw new ArgumentException("More inputs than slots.");
        }

        int T = 0, R = 0;
        Span<bool> seen = stackalloc bool[SlotCount];
        for (int u = 0; u < inputs.Count; u++)
        {
            var inp = inputs[u];
            Check(inp.Slot);
            if (seen[inp.Slot])
            {
                throw new ArgumentException($"Slot {inp.Slot} appears twice in one step.");
            }

            seen[inp.Slot] = true;
            int n = inp.Tokens.Length;
            if (n == 0)
            {
                throw new ArgumentException($"Empty input for slot {inp.Slot}.");
            }

            if (T + n > MaxBatchTokens)
            {
                throw new ArgumentException($"Step has more than {MaxBatchTokens} rows.");
            }

            int pos0 = _length[inp.Slot];
            if (pos0 + n > ContextLength)
            {
                throw new InvalidOperationException($"Context of slot {inp.Slot} full ({pos0} + {n} > {ContextLength}).");
            }

            inp.Tokens.Span.CopyTo(_tokBuf.AsSpan(T));
            for (int i = 0; i < n; i++)
            {
                int row = T + i, pos = pos0 + i;
                _rowBuf[row * 4] = inp.Slot;
                _rowBuf[row * 4 + 1] = pos;
                _rowBuf[row * 4 + 2] = n;
                _rowBuf[row * 4 + 3] = u;
                // Text rows use the same position on the temporal/height/width axes (M-RoPE), extra axes 0.
                for (int a = 0; a < _posAxes; a++)
                {
                    _posBuf[row * _posAxes + a] = a < 3 ? pos : 0;
                }
            }
            _runBuf[u * 4] = inp.Slot;
            _runBuf[u * 4 + 1] = T;
            _runBuf[u * 4 + 2] = n;
            _runBuf[u * 4 + 3] = pos0;
            if (inp.WantLogits)
            {
                _outBuf[R++] = T + n - 1;
            }

            T += n;
        }
        bool anyLogits = R > 0;
        if (!anyLogits)
        {
            _outBuf[R++] = T - 1;   // the graph needs at least one output row
        }

        _compiled.SetInput(_tokens, _tokBuf.AsSpan(0, T));
        if (_positions is not null)
        {
            _compiled.SetInput<int>(_positions, _posBuf.AsSpan(0, T * _posAxes));
        }

        _compiled.SetInput<int>(_rows, _rowBuf.AsSpan(0, T * 4));
        _compiled.SetInput<int>(_runs, _runBuf.AsSpan(0, inputs.Count * 4));
        if (_outputRows is not null)
        {
            _compiled.SetInput<int>(_outputRows, _outBuf.AsSpan(0, R));
        }

        _run.Symbols[WellKnown.TokenSymbol] = T;
        _run.Symbols["R"] = R;
        _run.Symbols[WellKnown.RunSymbol] = inputs.Count;
        _compiled.Run(_run);
        foreach (var inp in inputs)
        {
            _length[inp.Slot] += inp.Tokens.Length;
        }

        if (!anyLogits)
        {
            return res;
        }

        _compiled.GetOutput<float>("logits", _logits.AsSpan(0, R * VocabSize));
        for (int u = 0, r = 0; u < inputs.Count; u++)
        {
            if (!inputs[u].WantLogits)
            {
                continue;
            }

            var dst = _results[u];
            _logits.AsSpan(r++ * VocabSize, VocabSize).CopyTo(dst);
            res[u] = dst;
        }
        return res;
    }

    private void Check(int slot) { if ((uint)slot >= (uint)SlotCount)
        {
            throw new ArgumentOutOfRangeException(nameof(slot));
        }
    }

    public void Dispose() => _compiled.Dispose();
}
