using System.Text;
using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Deterministic multi-slot decoder: the next token of a slot is a pure function of its context (<see cref="Next"/>), the logits
/// are a one-hot spike on it. Records every step so tests can assert on batching. With <c>kvBlockSize</c> it also keeps the
/// block accounting of a paged KV cache (reference-counted blocks, shared prefixes) and fails a step that needs more blocks
/// than are free.
/// </summary>
public sealed class FakeBatchDecoder : IBatchDecoder
{
    public const int Marker = 1;   // prompts end with this token; scripted replies are indexed relative to it
    private readonly List<int>[] _ctx;
    private readonly float[] _logits;
    private readonly List<int>[] _blocks;
    private readonly int[] _refs;
    private readonly Stack<int> _free = new();

    public FakeBatchDecoder(int slots, int contextLength = 4096, int maxBatchTokens = 64, bool batches = true, bool canTruncate = true, int maxLogitsRows = 0,
        int kvBlockSize = 0, int kvBlocks = 0)
    {
        KvBlockSize = kvBlockSize;
        _blocks = [.. Enumerable.Range(0, slots).Select(_ => new List<int>())];
        _refs = new int[kvBlocks];
        for (int b = kvBlocks - 1; b >= 0; b--)
        {
            _free.Push(b);
        }

        SlotCount = slots;
        ContextLength = contextLength;
        MaxBatchTokens = maxBatchTokens;
        BatchesSequences = batches;
        CanTruncate = canTruncate;
        _ctx = [.. Enumerable.Range(0, slots).Select(_ => new List<int>())];
        MaxLogitsRows = Math.Max(slots, maxLogitsRows);
        _logits = new float[MaxLogitsRows * VocabSize];
        Next = ctx => (ctx[^1] + 1) % 256;
    }

    public int SlotCount { get; }
    public int ContextLength { get; }
    public int MaxBatchTokens { get; }
    public int MaxLogitsRows { get; }
    public int VocabSize => 260;
    public bool BatchesSequences { get; }
    public bool CanTruncate { get; }
    public Func<IReadOnlyList<int>, int> Next { get; set; }
    /// <summary>Logits of the tokens other than the scripted one (default: all -10); decides what a token mask falls back to.</summary>
    public float[]? Bias { get; set; }
    public TimeSpan StepDelay { get; set; }
    public int FailAfterSteps { get; set; } = -1;
    public List<(int Slot, int Rows, bool Logits)[]> Steps { get; } = [];
    public List<int[]> LogitsRowsPerStep { get; } = [];
    public List<(int Slot, int Length)> Truncations { get; } = [];
    public List<int> Resets { get; } = [];

    /// <summary>Replies with <paramref name="reply"/> (then end-of-generation) after a prompt ending in <see cref="Marker"/>; repeats if <paramref name="loop"/>.</summary>
    public void Script(string reply, bool loop = false)
    {
        var bytes = Encoding.UTF8.GetBytes(reply);
        Next = ctx =>
        {
            int m = -1;
            for (int i = ctx.Count - 1; i >= 0; i--)
            {
                if (ctx[i] == Marker)
                {
                    m = i;
                    break;
                }
            }

            int k = m < 0 ? 0 : ctx.Count - m - 1;
            if (loop && bytes.Length > 0)
            {
                k %= bytes.Length;
            }

            return k < bytes.Length ? bytes[k] : FakeTokenizer.Eos;
        };
    }

    public int KvBlockSize { get; }
    public int KvBlocksTotal => _refs.Length;
    public int KvBlocksFree => KvBlockSize > 0 ? _free.Count : int.MaxValue;
    /// <summary>Most blocks held at once.</summary>
    public int PeakBlocksHeld { get; private set; }
    public int KvBlocksNeeded(int slot, int tokens) => KvBlockSize == 0 ? 0 : Math.Max(0, BlocksFor(_ctx[slot].Count + tokens) - _blocks[slot].Count);
    public int KvBlocksReleasable(int slot) => _blocks[slot].Count(b => _refs[b] == 1);

    public int ShareKvPrefix(int slot, int source, int tokens)
    {
        if (KvBlockSize == 0)
        {
            return 0;
        }

        if (_ctx[slot].Count != 0)
        {
            throw new InvalidOperationException("slot not empty");
        }

        int n = Math.Min(tokens, _ctx[source].Count) / KvBlockSize;
        for (int j = 0; j < n; j++)
        {
            _refs[_blocks[source][j]]++;
            _blocks[slot].Add(_blocks[source][j]);
        }

        _ctx[slot].AddRange(_ctx[source].Take(n * KvBlockSize));
        return n * KvBlockSize;
    }

    private int BlocksFor(int tokens) => (tokens + KvBlockSize - 1) / KvBlockSize;

    private void Release(int slot, int keep)
    {
        var list = _blocks[slot];
        for (int j = list.Count - 1; j >= keep; j--)
        {
            if (--_refs[list[j]] == 0)
            {
                _free.Push(list[j]);
            }

            list.RemoveAt(j);
        }
    }

    public int Length(int slot) => _ctx[slot].Count;
    public void Reset(int slot)
    {
        _ctx[slot].Clear();
        Release(slot, 0);
        lock (Resets)
        {
            Resets.Add(slot);
        }
    }
    public void Truncate(int slot, int length)
    {
        if (!CanTruncate)
        {
            throw new NotSupportedException();
        }

        if (KvBlockSize > 0)
        {
            int keep = BlocksFor(length);
            if (length % KvBlockSize != 0 && _refs[_blocks[slot][keep - 1]] > 1)
            {
                keep--;
                length = keep * KvBlockSize;
            }

            Release(slot, keep);
        }

        _ctx[slot].RemoveRange(length, _ctx[slot].Count - length);
        lock (Truncations)
        {
            Truncations.Add((slot, length));
        }
    }
    public IReadOnlyList<int> Context(int slot) => _ctx[slot];

    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs)
    {
        if (inputs.Select(i => i.Slot).Distinct().Count() != inputs.Count)
        {
            throw new InvalidOperationException("two inputs for one slot");
        }

        if (inputs.Sum(i => i.Tokens.Length) > MaxBatchTokens)
        {
            throw new InvalidOperationException("batch too large");
        }

        if (!BatchesSequences && inputs.Count > 1)
        {
            /* executed one after another;
            still recorded as one step */
        }
        lock (Steps)
        {
            if (FailAfterSteps >= 0 && Steps.Count >= FailAfterSteps)
            {
                throw new InvalidOperationException("injected failure");
            }

            Steps.Add([.. inputs.Select(i => (i.Slot, i.Tokens.Length, i.WantLogits))]);
            LogitsRowsPerStep.Add([.. inputs.Select(i => i.OutputRows)]);
        }
        if (StepDelay > TimeSpan.Zero)
        {
            Thread.Sleep(StepDelay);
        }

        if (inputs.Sum(i => i.OutputRows) > MaxLogitsRows)
        {
            throw new InvalidOperationException("too many logits rows");
        }

        if (KvBlockSize > 0)
        {
            int need = inputs.Sum(i => KvBlocksNeeded(i.Slot, i.Tokens.Length));
            if (need > _free.Count)
            {
                throw new InvalidOperationException($"KV blocks exhausted: need {need}, {_free.Count} free");
            }

            foreach (var i in inputs)
            {
                for (int n = BlocksFor(_ctx[i.Slot].Count + i.Tokens.Length); _blocks[i.Slot].Count < n;)
                {
                    int b = _free.Pop();
                    _refs[b] = 1;
                    _blocks[i.Slot].Add(b);
                }
            }

            PeakBlocksHeld = Math.Max(PeakBlocksHeld, KvBlocksTotal - _free.Count);
        }

        var res = new Memory<float>[inputs.Count];
        int r = 0;
        for (int i = 0; i < inputs.Count; i++)
        {
            var c = _ctx[inputs[i].Slot];
            var toks = inputs[i].Tokens.ToArray();
            if (c.Count + toks.Length > ContextLength)
            {
                throw new InvalidOperationException("context overflow");
            }

            int rows = inputs[i].OutputRows;
            res[i] = _logits.AsMemory(r * VocabSize, rows * VocabSize);
            for (int k = 0; k < toks.Length; k++)
            {
                c.Add(toks[k]);
                int row = k - (toks.Length - rows);
                if (row < 0)
                {
                    continue;
                }

                var l = _logits.AsSpan((r + row) * VocabSize, VocabSize);
                if (Bias is { } bias)
                {
                    bias.CopyTo(l);
                }
                else
                {
                    l.Fill(-10f);
                }

                l[Next(c)] = 10f;
            }

            r += rows;
        }
        return res;
    }

    public void Dispose() { }
}
