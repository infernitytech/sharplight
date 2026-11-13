using System.Runtime.InteropServices;
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
/// Create with <see cref="LlmModel.CreateBatchDecoder"/>. Not thread-safe.
/// </summary>
public sealed class MultiSequenceDecoder : IBatchDecoder, IDeviceSamplingDecoder
{
    private static readonly SamplingRow?[] s_noSampling = [];
    private readonly CompiledModel _compiled;
    private readonly Value _tokens, _rows, _runs;
    private readonly Value? _positions, _outputRows, _samplingParams;
    private readonly int _posAxes;
    private readonly int[] _length;
    private readonly RunParameters _run = new();
    private readonly int[] _tokBuf, _posBuf, _rowBuf, _outBuf;
    private readonly int[] _runBuf;
    private readonly float[] _logits;
    private readonly int[] _paramBuf, _sampledBuf;
    private readonly float[] _logprobBuf;
    private readonly Memory<float>[] _results;

    /// <inheritdoc/>
    public int SlotCount { get; }
    /// <inheritdoc/>
    public int ContextLength { get; }
    /// <inheritdoc/>
    public int MaxBatchTokens { get; }
    /// <inheritdoc/>
    public int VocabSize { get; }
    /// <inheritdoc/>
    public int MaxLogitsRows { get; }
    /// <inheritdoc/>
    public bool BatchesSequences => true;
    /// <inheritdoc/>
    public bool CanTruncate { get; }
    internal CompiledModel Compiled => _compiled;
    /// <inheritdoc/>
    bool IDeviceSamplingDecoder.SamplesOnDevice => _samplingParams is not null;
    /// <inheritdoc/>
    public long DeviceMemoryBytes => _compiled.AcceleratorMemoryBytes;

    /// <param name="compiled">A model compiled from <see cref="MultiSequence.Transform"/>'s graph with context length
    /// <paramref name="slots"/> · <paramref name="contextLength"/> and symbol bounds U = slots, R ≥ slots.</param>
    /// <param name="slots">Number of sequence slots.</param>
    /// <param name="contextLength">Context length of each slot.</param>
    /// <param name="hasRecurrentState">Whether a slot holds state other than KV rows (then contexts cannot be truncated).</param>
    /// <param name="maxLogitsRows">Bound of the <c>R</c> symbol the model was compiled with (logits rows per step).</param>
    internal MultiSequenceDecoder(CompiledModel compiled, int slots, int contextLength, bool hasRecurrentState, int maxLogitsRows)
    {
        _compiled = compiled;
        var g = compiled.Graph;
        _tokens = g.Input(InputRole.TokenIds);
        _rows = g.Input(InputRole.SequenceRows);
        _runs = g.Input(InputRole.SequenceRuns);
        _positions = g.TryInput(InputRole.Positions);
        _outputRows = g.TryInput(InputRole.OutputRows);
        _samplingParams = g.TryInput(InputRole.SamplingParameters);
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
        MaxLogitsRows = Math.Max(slots, maxLogitsRows);
        _outBuf = new int[MaxLogitsRows];
        _paramBuf = new int[MaxLogitsRows * SamplingRow.Width];
        _sampledBuf = new int[MaxLogitsRows];
        _logprobBuf = new float[MaxLogitsRows];
        _logits = new float[(long)MaxLogitsRows * VocabSize > int.MaxValue ? throw new NotSupportedException("Logits buffer too large.") : MaxLogitsRows * VocabSize];
        _results = new Memory<float>[slots];
    }

    /// <summary>
    /// Compiles <paramref name="graph"/> (single-sequence decoder) for <paramref name="slots"/> sequences, with a device sampling head
    /// when <paramref name="deviceSampling"/> and the graph allows one.
    /// </summary>
    internal static MultiSequenceDecoder Create(ComputeGraph graph, int slots, int contextLength, int maxBatchTokens, PlacementOptions placement, bool deviceSampling)
    {
        var ms = MultiSequence.Transform(graph, slots);
        var decoderGraph = deviceSampling && SamplingHead.CanAppend(ms.Graph) ? SamplingHead.Append(ms.Graph) : ms.Graph;
        var c = placement.Compile;
        int maxLogitsRows = (int)Math.Min(maxBatchTokens, Math.Max(slots, c.SymbolBounds.GetValueOrDefault("R", 1)));
        var bounds = new Dictionary<string, long>(c.SymbolBounds)
        {
            [WellKnown.RunSymbol] = slots,
            ["R"] = maxLogitsRows,
        };
        var compiled = CompiledModel.Compile(decoderGraph, new PlacementOptions
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
        return new MultiSequenceDecoder(compiled, slots, contextLength, recurrent, maxLogitsRows);
    }

    /// <inheritdoc/>
    public int Length(int slot)
    {
        Check(slot);
        return _length[slot];
    }

    /// <inheritdoc/>
    public void Reset(int slot)
    {
        Check(slot);
        _length[slot] = 0;
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">The model has recurrent state and <paramref name="length"/> is neither 0 nor the current length.</exception>
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

    /// <inheritdoc/>
    /// <exception cref="ArgumentException">Too many inputs, a slot twice, an empty input, more rows than <see cref="MaxBatchTokens"/>
    /// or more logits rows than <see cref="MaxLogitsRows"/>.</exception>
    /// <exception cref="InvalidOperationException">A slot's context would overflow.</exception>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs) => Step(inputs, s_noSampling, [], []);

    /// <inheritdoc/>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs, IReadOnlyList<SamplingRow?> sampling, Span<int> tokens, Span<float> logprobs)
    {
        int count = inputs.Count;
        if (count == 0)
        {
            return [];
        }

        if (count > SlotCount)
        {
            throw new ArgumentException("More inputs than slots.");
        }

        int T = 0;
        int R = 0;
        Span<bool> seen = stackalloc bool[SlotCount];
        for (int u = 0; u < count; u++)
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

            int rows = inp.OutputRows;
            if (R + rows > MaxLogitsRows)
            {
                throw new ArgumentException($"Step asks for more than {MaxLogitsRows} logits rows.");
            }

            inp.Tokens.Span.CopyTo(_tokBuf.AsSpan(T));
            for (int i = 0; i < n; i++)
            {
                int row = T + i;
                int pos = pos0 + i;
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
            for (int i = n - rows; i < n; i++)
            {
                _outBuf[R++] = T + i;
            }

            T += n;
        }
        bool anyLogits = R > 0;
        if (!anyLogits)
        {
            _outBuf[R++] = T - 1;   // the graph needs at least one output row
        }

        bool needLogits = SetSamplingRows(inputs, sampling, R, anyLogits, out bool anySampled);

        _compiled.SetInput(_tokens, _tokBuf.AsSpan(0, T));
        if (_positions is not null)
        {
            _compiled.SetInput<int>(_positions, _posBuf.AsSpan(0, T * _posAxes));
        }

        _compiled.SetInput<int>(_rows, _rowBuf.AsSpan(0, T * 4));
        _compiled.SetInput<int>(_runs, _runBuf.AsSpan(0, count * 4));
        if (_outputRows is not null)
        {
            _compiled.SetInput<int>(_outputRows, _outBuf.AsSpan(0, R));
        }

        _run.Symbols[WellKnown.TokenSymbol] = T;
        _run.Symbols["R"] = R;
        _run.Symbols[WellKnown.RunSymbol] = count;
        _run.SkipOutputs.Clear();
        if (!needLogits)
        {
            _run.SkipOutputs.Add("logits");
        }

        _compiled.Run(_run);
        for (int u = 0; u < count; u++)
        {
            _length[inputs[u].Slot] += inputs[u].Tokens.Length;
        }

        if (needLogits)
        {
            // The per-input results are slices of one buffer (no per-sequence copies).
            ReadLogits(R);
        }

        if (anySampled)
        {
            _compiled.GetOutput<int>(SamplingHead.TokensOutput, _sampledBuf.AsSpan(0, R));
            _compiled.GetOutput<float>(SamplingHead.LogprobsOutput, _logprobBuf.AsSpan(0, R));
        }

        for (int u = 0, r = 0; u < count; u++)
        {
            int rows = inputs[u].OutputRows;
            if (rows == 1 && u < sampling.Count && sampling[u] is not null)
            {
                tokens[u] = _sampledBuf[r];
                logprobs[u] = _logprobBuf[r];
                _results[u] = Memory<float>.Empty;
            }
            else
            {
                _results[u] = rows == 0 ? Memory<float>.Empty : _logits.AsMemory(r * VocabSize, rows * VocabSize);
            }

            r += rows;
        }
        return new ArraySegment<Memory<float>>(_results, 0, count);
    }

    /// <summary>Downloads <paramref name="rows"/> logits rows; large batches are read by several threads (memory-bandwidth bound).</summary>
    private void ReadLogits(int rows)
    {
        const long ParallelReadBytes = 4L << 20;
        long rowBytes = (long)VocabSize * sizeof(float);
        int parts = (int)Math.Min(Math.Min(rows, Environment.ProcessorCount / 2), rows * rowBytes / ParallelReadBytes + 1);
        if (parts <= 1)
        {
            _compiled.GetOutput<float>("logits", _logits.AsSpan(0, rows * VocabSize));
            return;
        }

        Parallel.For(0, parts, part =>
        {
            int first = rows * part / parts;
            int end = rows * (part + 1) / parts;
            var destination = MemoryMarshal.AsBytes(_logits.AsSpan(first * VocabSize, (end - first) * VocabSize));
            _compiled.GetOutput("logits", first * rowBytes, destination);
        });
    }

    /// <summary>
    /// Fills the sampling parameters input (one row per output row; rows sampled on the CPU are skipped). Returns whether any output
    /// row still needs its logits downloaded.
    /// </summary>
    private bool SetSamplingRows(IReadOnlyList<SequenceInput> inputs, IReadOnlyList<SamplingRow?> sampling, int rows, bool anyLogits, out bool anySampled)
    {
        anySampled = false;
        if (_samplingParams is null)
        {
            if (sampling.Any(s => s is not null))
            {
                throw new InvalidOperationException("This decoder was compiled without a device sampling head.");
            }

            return anyLogits;
        }

        bool needLogits = false;
        int r = 0;
        for (int u = 0; u < inputs.Count; u++)
        {
            int outputRows = inputs[u].OutputRows;
            var row = u < sampling.Count ? sampling[u] : null;
            if (row is not null && outputRows != 1)
            {
                throw new ArgumentException($"Input {u} is sampled on the device but returns {outputRows} logits rows (device sampling needs exactly one).");
            }

            for (int i = 0; i < outputRows; i++)
            {
                var dst = _paramBuf.AsSpan(r++ * SamplingRow.Width, SamplingRow.Width);
                if (row is { } sampled)
                {
                    sampled.WriteTo(dst);
                    anySampled = true;
                }
                else
                {
                    SamplingRow.WriteSkip(dst);
                    needLogits = true;
                }
            }
        }

        for (; r < rows; r++)
        {
            SamplingRow.WriteSkip(_paramBuf.AsSpan(r * SamplingRow.Width, SamplingRow.Width));
        }

        _compiled.SetInput<int>(_samplingParams, _paramBuf.AsSpan(0, rows * SamplingRow.Width));
        return needLogits;
    }

    private void Check(int slot)
    {
        if ((uint)slot >= (uint)SlotCount)
        {
            throw new ArgumentOutOfRangeException(nameof(slot));
        }
    }

    /// <summary>Releases the compiled model and its device memory.</summary>
    public void Dispose() => _compiled.Dispose();
}
