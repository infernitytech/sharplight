using System.Text;
using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Deterministic multi-slot decoder: the next token of a slot is a pure function of its context (<see cref="Next"/>), the logits
/// are a one-hot spike on it. Records every step so tests can assert on batching. With <c>kvBlockSize</c> it also keeps the
/// block accounting of a paged KV cache (the engine's <see cref="KvBlockCache"/>: reference-counted blocks, prefix caching by content,
/// recurrent-state snapshots with <c>stateSnapshots</c>) and fails a step that needs more blocks than are free. Attached blocks bring
/// back the tokens they were computed from, so a wrongly attached block shows up as a wrong context.
/// </summary>
public sealed class FakeBatchDecoder : IBatchDecoder
{
    public const int Marker = 1;   // prompts end with this token; scripted replies are indexed relative to it
    private readonly List<int>[] _ctx;
    private readonly float[] _logits;
    private readonly KvBlockCache? _kv;
    private readonly int[][] _blockTokens;   // tokens each physical block was computed from
    private readonly List<int>[] _snapshotContext;   // per snapshot row: the context its "state" belongs to (checked on restore)

    public FakeBatchDecoder(int slots, int contextLength = 4096, int maxBatchTokens = 64, bool batches = true, bool canTruncate = true, int maxLogitsRows = 0,
        int kvBlockSize = 0, int kvBlocks = 0, int stateSnapshots = 0)
    {
        KvBlockSize = kvBlockSize;
        _blockTokens = new int[kvBlocks][];
        _snapshotContext = [.. Enumerable.Range(0, stateSnapshots).Select(_ => new List<int>())];
        if (kvBlockSize > 0)
        {
            _kv = new KvBlockCache(slots, kvBlockSize, kvBlocks, (contextLength + kvBlockSize - 1) / kvBlockSize, !canTruncate, stateSnapshots, CopyState);
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
    public int KvBlocksTotal => _kv?.Total ?? 0;
    public int KvBlocksFree => _kv?.Free ?? int.MaxValue;
    public int KvBlocksCached => _kv?.Cached ?? 0;
    public long KvBlocksEvicted => _kv?.Evictions ?? 0;
    public bool CachesKvBlocks => _kv?.CachesBlocks ?? false;
    public int KvStateSnapshots => _kv?.SnapshotCapacity ?? 0;
    /// <summary>Most blocks held at once.</summary>
    public int PeakBlocksHeld { get; private set; }
    /// <summary>Recurrent-state snapshots restored (attached blocks of a model with recurrent state).</summary>
    public int SnapshotRestores { get; private set; }
    public int KvBlocksNeeded(int slot, int tokens) => _kv?.Needed(slot, _ctx[slot].Count, tokens) ?? 0;
    public int KvBlocksReleasable(int slot) => _kv?.Releasable(slot) ?? 0;

    public int FindKvBlocks(ReadOnlySpan<UInt128> hashes, out int inUse)
    {
        inUse = 0;
        return _kv?.Find(hashes, out inUse) ?? 0;
    }

    public int AttachKvBlocks(int slot, ReadOnlySpan<UInt128> hashes)
    {
        if (_kv is null)
        {
            return 0;
        }

        int before = _kv.Blocks(slot).Count;
        int n = _kv.Attach(slot, _ctx[slot].Count, hashes);
        for (int j = before; j < before + n; j++)
        {
            _ctx[slot].AddRange(_blockTokens[_kv.Blocks(slot)[j]]);
        }

        if (_restoring is { } expected)
        {
            _restoring = null;
            if (!_ctx[slot].SequenceEqual(expected))
            {
                throw new InvalidOperationException("The restored recurrent state belongs to another context than the attached blocks.");
            }
        }

        return n * KvBlockSize;
    }

    public void PublishKvBlocks(int slot, int firstBlock, ReadOnlySpan<UInt128> hashes)
    {
        int full = _ctx[slot].Count / Math.Max(1, KvBlockSize);
        if (_kv is { CachesBlocks: true } && firstBlock < full)
        {
            _kv.Publish(slot, firstBlock, hashes[..Math.Min(hashes.Length, full - firstBlock)]);
        }
    }

    public bool SaveKvStateSnapshot(int slot, UInt128 hash) => _kv is not null && _ctx[slot].Count % KvBlockSize == 0 && _kv.SaveSnapshot(slot, hash);

    /// <summary>"Copies recurrent state" between slot rows and snapshot rows: the fake state is the context it belongs to.</summary>
    private bool CopyState(int from, int to)
    {
        int slots = SlotCount;
        if (to >= slots)
        {
            _snapshotContext[to - slots].Clear();
            _snapshotContext[to - slots].AddRange(_ctx[from]);
        }
        else
        {
            // Restoring: the slot's context must be exactly the blocks before the snapshot boundary, which Attach appends next.
            SnapshotRestores++;
            _restoring = _snapshotContext[from - slots];
        }

        return true;
    }

    private List<int>? _restoring;

    public int Length(int slot) => _ctx[slot].Count;
    public void Reset(int slot)
    {
        _ctx[slot].Clear();
        _kv?.Release(slot, 0);
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

        if (_kv is not null)
        {
            length = _kv.Truncate(slot, length);
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

        if (_kv is not null)
        {
            int need = inputs.Sum(i => KvBlocksNeeded(i.Slot, i.Tokens.Length));
            if (need > _kv.Free)
            {
                throw new InvalidOperationException($"KV blocks exhausted: need {need}, {_kv.Free} free");
            }

            foreach (var i in inputs)
            {
                _kv.Grow(i.Slot, _ctx[i.Slot].Count + i.Tokens.Length);
            }

            PeakBlocksHeld = Math.Max(PeakBlocksHeld, KvBlocksTotal - _kv.Free);
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
            if (_kv is not null)
            {
                // Full blocks remember the tokens they hold (a block is never rewritten while published or shared).
                var blocks = _kv.Blocks(inputs[i].Slot);
                for (int j = (c.Count - toks.Length) / KvBlockSize; j < c.Count / KvBlockSize; j++)
                {
                    _blockTokens[blocks[j]] = [.. c.Skip(j * KvBlockSize).Take(KvBlockSize)];
                }
            }
        }
        return res;
    }

    public void Dispose()
    {
    }
}
