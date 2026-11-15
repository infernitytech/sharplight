namespace Sharplight.Llm.Structured;

/// <summary>
/// Per-request state of a <see cref="GrammarConstraint"/>: the grammar state (or the free phase before the trigger), the
/// history for <see cref="Rollback"/>, and the per-state cache of allowed-token masks.
/// </summary>
internal sealed class GrammarMatcher : IConstraintMatcher
{
    private readonly GrammarConstraint _constraint;
    private readonly TokenTrie _trie;
    private readonly TokenMaskCache _cache;
    private readonly List<(int Start, int End)> _uncertain = [];
    private readonly int[] _specialTokens;
    private readonly byte[]? _trigger;
    private readonly GrammarRuntime _runtime;
    private readonly List<(GrammarState? State, byte[] Tail)> _history = [];
    private readonly Stack<(int Node, GrammarState State)> _walk = new();
    private GrammarState? _state;
    /// <summary>In the free phase: the last output bytes that could still be the start of the trigger.</summary>
    private byte[] _tail = [];
    private int _cachedMasks;

    public GrammarMatcher(GrammarConstraint constraint, TokenTrie trie, TokenMaskCache cache, int[] specialTokens, byte[]? trigger)
    {
        _constraint = constraint;
        _trie = trie;
        _cache = cache;
        _specialTokens = specialTokens;
        _trigger = trigger;
        _runtime = new GrammarRuntime(constraint.Grammar);
        _state = trigger is null ? _runtime.Initial : null;
    }

    private bool Free => _state is null;

    public bool IsComplete => Free ? _constraint.Options.TriggerOptional : _state!.Accepting;

    public bool IsTerminated => !Free && _state!.Accepting && _state.Allowed.IsEmpty;

    public void FillAllowed(TokenMask mask)
    {
        if (_state is not { } state)
        {
            mask.AllowAll();
            return;
        }

        var words = mask.WritableWords;
        if (state.CachedMask is { } cached && cached.Length == words.Length)
        {
            cached.CopyTo(words);
        }
        else
        {
            if (!FromCache(state, mask))
            {
                mask.Clear();
                Walk(state, mask);
            }

            if (_cachedMasks < _constraint.Options.MaskCacheSize)
            {
                state.CachedMask = words.ToArray();
                _cachedMasks++;
            }
        }

        foreach (int id in _specialTokens)
        {
            if (_runtime.Advance(state, _constraint.Tokenizer.TokenBytes(id)) != _runtime.Dead)
            {
                mask.Allow(id);
            }
        }
    }

    /// <summary>
    /// Combines the shared per-stack-top entries of <paramref name="state"/>'s stacks and checks their context-dependent
    /// tokens against the full state. False when the shared cache is full (the caller walks the trie instead).
    /// </summary>
    private bool FromCache(GrammarState state, TokenMask mask)
    {
        mask.Clear();
        _uncertain.Clear();
        var words = mask.WritableWords;
        foreach (var stack in state.Stacks)
        {
            if (stack.Position < 0)
            {
                continue;
            }

            if (_cache.Get(stack) is not { } entry)
            {
                return false;
            }

            if (entry.Words is { } w)
            {
                Or(words, w);
            }
            else
            {
                foreach (int t in entry.Tokens!)
                {
                    words[t >> 5] |= 1u << (t & 31);
                }
            }

            _uncertain.AddRange(entry.Uncertain);
        }

        var tokens = _trie.Tokens;
        var tokenizer = _constraint.Tokenizer;
        foreach (var (start, end) in _uncertain)
        {
            for (int i = start; i < end; i++)
            {
                int t = tokens[i];
                if ((words[t >> 5] & (1u << (t & 31))) == 0 && _runtime.Advance(state, tokenizer.TokenBytes(t)) != _runtime.Dead)
                {
                    words[t >> 5] |= 1u << (t & 31);
                }
            }
        }

        return true;
    }

    private static void Or(Span<uint> target, ReadOnlySpan<uint> source)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int width = System.Numerics.Vector<uint>.Count;
            for (; i + width <= target.Length; i += width)
            {
                var v = new System.Numerics.Vector<uint>(target[i..]) | new System.Numerics.Vector<uint>(source[i..]);
                v.CopyTo(target[i..]);
            }
        }

        for (; i < target.Length; i++)
        {
            target[i] |= source[i];
        }
    }

    /// <summary>Depth-first walk of the vocabulary trie, descending only along bytes the grammar accepts.</summary>
    private void Walk(GrammarState start, TokenMask mask)
    {
        var childStart = _trie.ChildStart;
        var childByte = _trie.ChildByte;
        var childNode = _trie.ChildNode;
        var tokenStart = _trie.TokenStart;
        var tokens = _trie.Tokens;
        var dead = _runtime.Dead;
        _walk.Clear();
        _walk.Push((0, start));
        while (_walk.TryPop(out var top))
        {
            var state = top.State;
            var allowed = state.Allowed;
            for (int i = childStart[top.Node]; i < childStart[top.Node + 1]; i++)
            {
                byte b = childByte[i];
                if (!allowed.Contains(b))
                {
                    continue;
                }

                var next = _runtime.Advance(state, b);
                if (next == dead)
                {
                    continue;
                }

                int child = childNode[i];
                for (int t = tokenStart[child]; t < tokenStart[child + 1]; t++)
                {
                    mask.Allow(tokens[t]);
                }

                if (childStart[child] < childStart[child + 1] && !next.Allowed.IsEmpty)
                {
                    _walk.Push((child, next));
                }
            }
        }
    }

    public bool Accept(int token)
    {
        byte[] bytes = _constraint.Tokenizer.TokenBytes(token);
        if (_state is not { } state)
        {
            _history.Add((null, _tail));
            AcceptFree(bytes);
            return true;
        }

        var next = _runtime.Advance(state, bytes);
        if (bytes.Length == 0 || next == _runtime.Dead)
        {
            return false;
        }

        _history.Add((state, _tail));
        _state = next;
        return true;
    }

    /// <summary>Free phase: looks for the trigger across the token boundary and starts the grammar after it.</summary>
    private void AcceptFree(byte[] bytes)
    {
        var trigger = _trigger!;
        byte[] text = [.. _tail, .. bytes];
        int at = text.AsSpan().IndexOf(trigger);
        if (at < 0)
        {
            int keep = Math.Min(trigger.Length - 1, text.Length);
            _tail = text[^keep..];
            return;
        }

        var rest = text.AsSpan(at + trigger.Length);
        var state = _runtime.Advance(_runtime.Initial, rest);
        // Text the model glued to the trigger that the grammar rejects is left unconstrained rather than failing the request.
        _state = state == _runtime.Dead ? _runtime.Initial : state;
        _tail = [];
    }

    public void Rollback(int count)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, _history.Count);
        if (count <= 0)
        {
            return;
        }

        var (state, tail) = _history[^count];
        _history.RemoveRange(_history.Count - count, count);
        _state = state;
        _tail = tail;
    }
}
