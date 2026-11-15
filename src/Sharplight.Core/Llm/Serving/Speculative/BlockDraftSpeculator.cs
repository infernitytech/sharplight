namespace Sharplight.Llm.Serving.Speculative;

/// <summary>
/// Block speculation: a block drafter (<see cref="Models.WellKnown.DraftBlockSize"/>, DFlash) proposes all drafts of a step in one pass of
/// its draft graph. Create it with <see cref="LlmModel.CreateMtpSpeculator"/>.
/// </summary>
/// <remarks>
/// The drafter runs as its own multi-sequence decoder with one slot per engine slot and its own KV cache, which holds rows built from the
/// target's features of every verified position. After each target step the engine hands over the kept rows (<see cref="Observe"/>);
/// <see cref="Propose"/> feeds the new rows of every drafting sequence together with its block, the pending token followed by
/// <see cref="MaxDraftTokens"/> mask tokens at the next positions, all sequences batched, and reads one draft per mask row: the drafter's
/// greedy token, or the best path through its candidate lattice (<see cref="Models.WellKnown.DraftLatticeCandidates"/>). The block rows are
/// cut from the drafter's cache afterwards. Drafts are greedy, which the engine's verification keeps distribution-exact for any sampling
/// (see <see cref="ISpeculator"/>).
/// </remarks>
public sealed class BlockDraftSpeculator : ISpeculator
{
    private readonly MultiSequenceDecoder _draft;
    private readonly int _features;   // width of the target features at the start of every input row
    private readonly int _width;      // width of the drafter's hidden-state input rows
    private readonly int _convTaps;
    private readonly int _lattice;    // candidates per lattice row, 0 = greedy tokens
    private readonly int _mask;
    private readonly int[]? _tokenMap;
    private readonly MtpRowBuffer[] _buffers;
    private readonly int[] _lastSuccessor;
    private readonly List<SpeculationRequest> _active = [];
    private readonly List<SpeculationRequest> _group = [];
    private readonly List<SequenceInput> _inputs = [];
    private float[] _hidden;
    private int[] _tokens;

    internal BlockDraftSpeculator(MultiSequenceDecoder draft, int draftTokens, int blockSize, int maskToken, int convTaps, int latticeCandidates,
        int[]? tokenMap = null)
    {
        _draft = draft;
        _width = draft.HiddenInputSize;
        _convTaps = convTaps;
        _lattice = latticeCandidates;
        _features = _width - 1 - (2 * convTaps) - (latticeCandidates > 0 ? 2 : 0);
        if (_features <= 0)
        {
            throw new ArgumentException("The draft decoder's hidden-states input is narrower than its block columns.", nameof(draft));
        }

        if (draft.VocabSize != (latticeCandidates > 0 ? latticeCandidates + (latticeCandidates * latticeCandidates) : 1))
        {
            throw new ArgumentException("The draft decoder's logits rows do not match its block drafting form.", nameof(draft));
        }

        _mask = maskToken;
        _tokenMap = tokenMap;
        MaxDraftTokens = Math.Clamp(draftTokens, 1, blockSize - 1);
        if (BlockRows > draft.MaxBatchTokens)
        {
            throw new ArgumentException($"A block of {BlockRows} rows exceeds the draft decoder's {draft.MaxBatchTokens} rows per step.", nameof(draft));
        }

        _buffers = [.. Enumerable.Range(0, draft.SlotCount).Select(_ => new MtpRowBuffer(_features))];
        _lastSuccessor = new int[draft.SlotCount];
        _hidden = new float[_width * Math.Max(draft.SlotCount * BlockRows, 64)];
        _tokens = new int[Math.Max(draft.SlotCount * BlockRows, 64)];
    }

    /// <inheritdoc/>
    public int MaxDraftTokens { get; }

    /// <summary>Rows of a block: the pending token and one mask row per draft.</summary>
    private int BlockRows => MaxDraftTokens + 1;

    /// <summary>Width of the target features the drafter reads (the target's draft hidden source).</summary>
    public int HiddenSize => _features;

    /// <inheritdoc/>
    public long DeviceMemoryBytes => _draft.DeviceMemoryBytes;

    /// <summary>The drafter's decoder (compiled draft graph).</summary>
    internal MultiSequenceDecoder Decoder => _draft;

    /// <inheritdoc/>
    public bool UsesHiddenStates => true;

    /// <inheritdoc/>
    public void Observe(int slot, int position, ReadOnlySpan<int> successors, ReadOnlySpan<float> hidden)
    {
        if (successors.Length == 0)
        {
            return;
        }

        var buf = _buffers[slot];
        buf.Append(position, successors, hidden);
        _lastSuccessor[slot] = successors[^1];
        // Long prompts: write the features now, in bounded chunks.
        if (buf.Count > _draft.MaxBatchTokens / 2)
        {
            FeedContext(slot);
        }
    }

    /// <inheritdoc/>
    public void Propose(IReadOnlyList<SpeculationRequest> requests)
    {
        try
        {
            ProposeAll(requests);
        }
        catch (InvalidOperationException)
        {
            // The drafter's paged cache ran out of blocks: start the sequences over; they propose again once new rows were observed.
            foreach (var r in requests)
            {
                r.Count = 0;
                Release(r.Slot);
            }
        }
    }

    private void ProposeAll(IReadOnlyList<SpeculationRequest> requests)
    {
        _active.Clear();
        foreach (var r in requests)
        {
            r.Count = 0;
            var buf = _buffers[r.Slot];
            // The observed rows must end exactly before the pending token, the last row's successor; the drafter's cache holds the rest.
            int ctx = r.Context.Count;
            if (r.MaxTokens <= 0 || buf.End != ctx - 1 || _lastSuccessor[r.Slot] != r.Context[ctx - 1]
                || (buf.Count == 0 && _draft.Length(r.Slot) != buf.End) || buf.End + BlockRows > _draft.ContextLength)
            {
                continue;
            }

            if (buf.Count + BlockRows > _draft.MaxBatchTokens)
            {
                FeedContext(r.Slot);
            }

            _active.Add(r);
        }

        // Steps of whole sequences (context rows and block) within the drafter's rows per step.
        int start = 0;
        while (start < _active.Count)
        {
            _group.Clear();
            int rows = 0;
            for (int u = start; u < _active.Count; u++)
            {
                int n = _buffers[_active[u].Slot].Count + BlockRows;
                if (_group.Count > 0 && (rows + n > _draft.MaxBatchTokens || (_group.Count + 1) * BlockRows > _draft.MaxLogitsRows))
                {
                    break;
                }

                _group.Add(_active[u]);
                rows += n;
            }

            DraftGroup(rows);
            start += _group.Count;
        }
    }

    /// <summary>One draft pass over the sequences of <see cref="_group"/> (<paramref name="rows"/> step rows in all).</summary>
    private void DraftGroup(int rows)
    {
        Ensure(rows);
        Array.Clear(_hidden, 0, rows * _width);
        _inputs.Clear();
        int t = 0;
        int logitsRow = 0;
        foreach (var r in _group)
        {
            var buf = _buffers[r.Slot];
            SyncLength(r.Slot, buf.Start);
            int first = t;
            t = ContextRows(buf, t);
            for (int i = 0; i < BlockRows; i++, t++)
            {
                _tokens[t] = i == 0 ? r.Context[^1] : _mask;
                var row = _hidden.AsSpan(t * _width, _width);
                for (int j = 1; j <= _convTaps && j <= i; j++)
                {
                    row[_features + 1 + (2 * (j - 1))] = t - j;
                    row[_features + 2 + (2 * (j - 1))] = 1f;
                }

                if (_lattice > 0)
                {
                    // The previous logits row (the pending token's own row for the first mask row).
                    row[_features + 1 + (2 * _convTaps)] = logitsRow + Math.Max(0, i - 1);
                    row[_features + 2 + (2 * _convTaps)] = i <= 1 ? 1f : 0f;
                }
            }

            _inputs.Add(new SequenceInput(r.Slot, _tokens.AsMemory(first, t - first), LogitsRows: BlockRows));
            logitsRow += BlockRows;
        }

        _draft.SetHiddenInput(_hidden.AsSpan(0, rows * _width));
        var outputs = _draft.Step(_inputs);
        for (int u = 0; u < _group.Count; u++)
        {
            var r = _group[u];
            var buf = _buffers[r.Slot];
            int pending = buf.End;
            ReadDrafts(r, outputs[u].Span);
            buf.RemoveFirst(buf.Count);
            _draft.Truncate(r.Slot, pending);
        }
    }

    /// <summary>Fills the context rows of <paramref name="buf"/> from step row <paramref name="t"/> on; returns the row after them.</summary>
    private int ContextRows(MtpRowBuffer buf, int t)
    {
        var tokens = buf.Tokens;
        var hidden = buf.Hidden;
        for (int i = 0; i < buf.Count; i++, t++)
        {
            _tokens[t] = tokens[i];
            var row = _hidden.AsSpan(t * _width, _width);
            hidden.Slice(i * _features, _features).CopyTo(row);
            row[_features] = 1f;
        }

        return t;
    }

    /// <summary>The drafts of one sequence from its block's logits rows (the pending token's row first).</summary>
    private void ReadDrafts(SpeculationRequest r, ReadOnlySpan<float> block)
    {
        int width = _draft.VocabSize;
        int count = Math.Min(MaxDraftTokens, r.MaxTokens);
        int predecessor = 0;
        for (int i = 1; i <= count; i++)
        {
            var row = block.Slice(i * width, width);
            int token;
            if (_lattice == 0)
            {
                token = (int)row[0];
            }
            else
            {
                var scores = row.Slice(_lattice + (predecessor * _lattice), _lattice);
                predecessor = 0;
                for (int c = 1; c < _lattice; c++)
                {
                    if (scores[c] > scores[predecessor])
                    {
                        predecessor = c;
                    }
                }

                token = (int)row[predecessor];
            }

            r.Draft[i - 1] = _tokenMap is null ? token : _tokenMap[token];
        }

        r.Count = count;
    }

    /// <inheritdoc/>
    public void Release(int slot)
    {
        _buffers[slot].Clear();
        _draft.Reset(slot);
    }

    /// <inheritdoc/>
    public void Dispose() => _draft.Dispose();

    /// <summary>Writes a sequence's buffered rows into the drafter's cache (no logits), in steps of bounded size.</summary>
    private void FeedContext(int slot)
    {
        var buf = _buffers[slot];
        while (buf.Count > 0)
        {
            int n = Math.Min(buf.Count, _draft.MaxBatchTokens);
            SyncLength(slot, buf.Start);
            Ensure(n);
            Array.Clear(_hidden, 0, n * _width);
            var tokens = buf.Tokens;
            var hidden = buf.Hidden;
            for (int i = 0; i < n; i++)
            {
                hidden.Slice(i * _features, _features).CopyTo(_hidden.AsSpan(i * _width));
                _hidden[(i * _width) + _features] = 1f;
            }

            _draft.SetHiddenInput(_hidden.AsSpan(0, n * _width));
            _draft.Step([new SequenceInput(slot, tokens[..n].ToArray(), WantLogits: false)]);
            buf.RemoveFirst(n);
        }
    }

    /// <summary>
    /// Moves the drafter's cache of <paramref name="slot"/> to end at <paramref name="position"/>: cut back after rejected rows, or skipped
    /// ahead over rows it never saw (e.g. a reused prompt prefix; those rows only make drafts less likely to match).
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

    private void Ensure(int rows)
    {
        if (_hidden.Length < rows * _width)
        {
            _hidden = new float[rows * _width];
        }

        if (_tokens.Length < rows)
        {
            _tokens = new int[rows];
        }
    }
}
