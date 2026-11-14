using Sharplight.Graph;

namespace Sharplight.Llm.Serving.Speculative;

/// <summary>
/// Multi-token-prediction speculation: the target model's own MTP / NextN head (<see cref="Models.WellKnown.DraftGraph"/>)
/// proposes drafts from the target's hidden states. Create it with <see cref="LlmModel.CreateMtpSpeculator"/>.
/// </summary>
/// <remarks>
/// The head runs as its own multi-sequence decoder with one slot per engine slot and its own KV cache. Its row at position
/// <c>i</c> combines the target's hidden state at <c>i</c> with the token at <c>i + 1</c> and predicts the token at <c>i + 2</c>.
/// After each target step the engine hands over the kept rows (<see cref="Observe"/>); <see cref="Propose"/> feeds them, all
/// sequences batched, which yields the first draft (after the pending token), then chains one row per further draft with the
/// head's own hidden output, and finally cuts the head's cache back to the target-verified rows. Drafts are greedy (the head's
/// argmax), which the engine's verification keeps distribution-exact for any sampling (see <see cref="ISpeculator"/>).
/// </remarks>
public sealed class MtpSpeculator : ISpeculator
{
    private readonly MultiSequenceDecoder _draft;
    private readonly int _hidden;
    private readonly MtpRowBuffer[] _buffers;
    private readonly int[] _verified;
    private readonly List<SpeculationRequest> _active = [];
    private readonly List<SequenceInput> _inputs = [];
    private readonly int[] _tokenScratch;
    private float[] _hiddenScratch;
    private readonly float[] _chain;
    // The head's greedy choice per input of a draft step: on the device when the head has a sampling head (only token ids come
    // back), else the arg-max of its downloaded logits.
    private readonly IDeviceSamplingDecoder? _deviceArgmax;
    private readonly SamplingRow?[] _greedyRows;
    private readonly int[] _argmax;
    private readonly float[] _argmaxLogprobs;

    internal MtpSpeculator(MultiSequenceDecoder draft, int maxDraftTokens)
    {
        _draft = draft;
        _hidden = draft.HiddenInputSize;
        if (_hidden == 0 || draft.HiddenSize != _hidden)
        {
            throw new ArgumentException("The draft decoder needs a hidden-states input and a hidden output of the same width.", nameof(draft));
        }

        MaxDraftTokens = Math.Max(1, maxDraftTokens);
        _draft.CaptureHiddenStates = true;
        _buffers = [.. Enumerable.Range(0, draft.SlotCount).Select(_ => new MtpRowBuffer(_hidden))];
        _verified = new int[draft.SlotCount];
        _tokenScratch = new int[draft.SlotCount];
        _hiddenScratch = new float[_hidden * Math.Max(draft.SlotCount, 16)];
        _chain = new float[_hidden * draft.SlotCount];
        _deviceArgmax = draft is IDeviceSamplingDecoder { SamplesOnDevice: true } d ? d : null;
        _greedyRows = [.. Enumerable.Repeat<SamplingRow?>(SamplingRow.Greedy(), draft.SlotCount)];
        _argmax = new int[draft.MaxLogitsRows];
        _argmaxLogprobs = new float[draft.MaxLogitsRows];
    }

    /// <summary>Device memory the draft head holds (its own weights, KV cache and buffers; weights shared with the decoder are not counted).</summary>
    public long DeviceMemoryBytes => _draft.DeviceMemoryBytes;

    /// <summary>Runs the head on <see cref="_inputs"/> (one logits row each) and leaves each input's greedy token in <see cref="_argmax"/>.</summary>
    private void DraftStep()
    {
        if (_deviceArgmax is not null)
        {
            _deviceArgmax.Step(_inputs, new ArraySegment<SamplingRow?>(_greedyRows, 0, _inputs.Count), _argmax, _argmaxLogprobs);
            return;
        }

        var logits = _draft.Step(_inputs);
        for (int u = 0; u < _inputs.Count; u++)
        {
            _argmax[u] = LogitsMath.ArgMax(logits[u].Span);
        }
    }

    /// <inheritdoc/>
    public int MaxDraftTokens { get; }

    /// <summary>Hidden size the head expects (the target's <see cref="IBatchDecoder.HiddenSize"/>).</summary>
    public int HiddenSize => _hidden;

    /// <summary>The head's decoder (compiled draft graph).</summary>
    internal MultiSequenceDecoder Decoder => _draft;

    /// <inheritdoc/>
    public bool UsesHiddenStates => true;

    /// <inheritdoc/>
    public void Observe(int slot, int position, ReadOnlySpan<int> successors, ReadOnlySpan<float> hidden)
    {
        var buf = _buffers[slot];
        buf.Append(position, successors, hidden);
        // Long prompts: feed all but the last row now (the last one yields the first draft later), in bounded chunks.
        if (buf.Count > _draft.MaxBatchTokens / 2)
        {
            FeedAllButLast(slot);
        }
    }

    /// <inheritdoc/>
    public void Propose(IReadOnlyList<SpeculationRequest> requests)
    {
        try
        {
            ProposeBatch(requests);
        }
        catch (InvalidOperationException)
        {
            // The head's paged cache ran out of blocks (it can hold more rows than the target after prefix reuse): start the
            // sequences over; they propose again once new rows were observed.
            foreach (var r in requests)
            {
                r.Count = 0;
                Release(r.Slot);
            }
        }
    }

    private void ProposeBatch(IReadOnlyList<SpeculationRequest> requests)
    {
        _active.Clear();
        foreach (var r in requests)
        {
            r.Count = 0;
            var buf = _buffers[r.Slot];
            // The buffer must end exactly before the pending token, which is its last row's successor.
            int ctx = r.Context.Count;
            if (r.MaxTokens <= 0 || buf.Count == 0 || buf.End != ctx - 1 || buf.Tokens[^1] != r.Context[ctx - 1])
            {
                continue;
            }

            _active.Add(r);
        }

        if (_active.Count == 0)
        {
            return;
        }

        int total = 0;
        foreach (var r in _active)
        {
            total += _buffers[r.Slot].Count;
        }

        if (total > _draft.MaxBatchTokens)
        {
            foreach (var r in _active)
            {
                FeedAllButLast(r.Slot);
            }

            total = _active.Count;
        }

        // First draft: every buffered row, logits at each sequence's last row.
        _inputs.Clear();
        EnsureHiddenScratch(total);
        int row = 0;
        foreach (var r in _active)
        {
            var buf = _buffers[r.Slot];
            SyncLength(r.Slot, buf.Start);
            _inputs.Add(new SequenceInput(r.Slot, buf.Tokens.ToArray()));
            buf.Hidden.CopyTo(_hiddenScratch.AsSpan(row * _hidden));
            row += buf.Count;
        }

        _draft.SetHiddenInput(_hiddenScratch.AsSpan(0, total * _hidden));
        DraftStep();
        for (int u = 0; u < _active.Count; u++)
        {
            var r = _active[u];
            var buf = _buffers[r.Slot];
            r.Draft[0] = _argmax[u];
            r.Count = 1;
            _draft.HiddenStates(u, buf.Count).Span[((buf.Count - 1) * _hidden)..].CopyTo(_chain.AsSpan(u * _hidden, _hidden));
            _verified[r.Slot] = buf.End;
            buf.Clear();
        }

        // Further drafts: one row per sequence, the previous draft token with the head's own hidden output.
        int limit = Math.Min(MaxDraftTokens, _active.Max(r => r.MaxTokens));
        for (int j = 1; j < limit; j++)
        {
            _inputs.Clear();
            int n = 0;
            for (int u = 0; u < _active.Count; u++)
            {
                var r = _active[u];
                if (r.MaxTokens <= j || r.Count != j)
                {
                    continue;
                }

                _tokenScratch[n] = r.Draft[j - 1];
                _inputs.Add(new SequenceInput(r.Slot, _tokenScratch.AsMemory(n, 1)));
                _chain.AsSpan(u * _hidden, _hidden).CopyTo(_hiddenScratch.AsSpan(n * _hidden));
                n++;
            }

            if (n == 0)
            {
                break;
            }

            _draft.SetHiddenInput(_hiddenScratch.AsSpan(0, n * _hidden));
            DraftStep();
            for (int u = 0, k = 0; u < _active.Count; u++)
            {
                var r = _active[u];
                if (r.MaxTokens <= j || r.Count != j)
                {
                    continue;
                }

                r.Draft[j] = _argmax[k];
                r.Count = j + 1;
                _draft.HiddenStates(k, 1).Span.CopyTo(_chain.AsSpan(u * _hidden, _hidden));
                k++;
            }
        }

        // The chained rows were the head's guesses: keep only rows built from the target's hidden states.
        foreach (var r in _active)
        {
            if (_draft.Length(r.Slot) > _verified[r.Slot])
            {
                _draft.Truncate(r.Slot, _verified[r.Slot]);
            }
        }
    }

    /// <inheritdoc/>
    public void Release(int slot)
    {
        _buffers[slot].Clear();
        _draft.Reset(slot);
    }

    /// <inheritdoc/>
    public void Dispose() => _draft.Dispose();

    /// <summary>Feeds a sequence's buffered rows except the last one to the head (no logits), in steps of bounded size.</summary>
    private void FeedAllButLast(int slot)
    {
        var buf = _buffers[slot];
        while (buf.Count > 1)
        {
            int n = Math.Min(buf.Count - 1, _draft.MaxBatchTokens);
            SyncLength(slot, buf.Start);
            EnsureHiddenScratch(n);
            buf.Hidden[..(n * _hidden)].CopyTo(_hiddenScratch);
            _draft.SetHiddenInput(_hiddenScratch.AsSpan(0, n * _hidden));
            _draft.Step([new SequenceInput(slot, buf.Tokens[..n].ToArray(), WantLogits: false)]);
            buf.RemoveFirst(n);
        }
    }

    /// <summary>
    /// Moves the head's cache of <paramref name="slot"/> to end at <paramref name="position"/>: cut back after rejected rows,
    /// or skipped ahead over rows it never saw (e.g. a reused prompt prefix; those rows only make drafts less likely to match).
    /// </summary>
    private void SyncLength(int slot, int position)
    {
        int length = _draft.Length(slot);
        if (position == length)
        {
            return;
        }

        if (position == 0)
        {
            _draft.Reset(slot);
        }
        else if (position < length)
        {
            _draft.Truncate(slot, position);
        }
        else
        {
            _draft.SetLength(slot, position);
        }
    }

    private void EnsureHiddenScratch(int rows)
    {
        if (_hiddenScratch.Length < rows * _hidden)
        {
            _hiddenScratch = new float[rows * _hidden];
        }
    }
}
