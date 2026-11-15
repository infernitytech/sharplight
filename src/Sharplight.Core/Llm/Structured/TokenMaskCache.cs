using System.Collections.Concurrent;

namespace Sharplight.Llm.Structured;

/// <summary>
/// Allowed tokens per parse-stack <em>top</em>, shared by all requests of a <see cref="GrammarConstraint"/> (in the spirit
/// of XGrammar's context-independent token cache). A stack is keyed by its top <see cref="Depth"/> frames only; walking the
/// vocabulary trie from that truncated stack splits the tokens into those accepted within the known frames (a bitmask,
/// valid in every context) and those whose bytes run past the bottom frame (a few token ranges, checked against the real
/// state at mask time). The same JSON string, number or literal position therefore costs one trie walk per grammar, not
/// one per context and request.
/// </summary>
/// <remarks>Thread-safe; entries are computed under a lock with a private runtime and published read-only.</remarks>
internal sealed class TokenMaskCache
{
    /// <summary>Frames that key an entry: enough to cover JSON's string / repetition / value nesting.</summary>
    public const int Depth = 3;
    /// <summary>Entries kept per grammar; beyond that, masks are computed per state without this cache.</summary>
    private const int MaxEntries = 4096;
    /// <summary>Entries with fewer allowed tokens store a list instead of a VocabSize-bit mask.</summary>
    private const int SparseLimit = 2048;

    private readonly TokenTrie _trie;
    private readonly GrammarRuntime _runtime;
    private readonly ConcurrentDictionary<string, TokenMaskEntry> _entries = new(StringComparer.Ordinal);
    private readonly Stack<(int Node, GrammarState State, bool Covered)> _walk = new();
    private readonly Lock _lock = new();

    public TokenMaskCache(Grammar grammar, TokenTrie trie)
    {
        _trie = trie;
        _runtime = new GrammarRuntime(grammar);
    }

    public int Count => _entries.Count;

    /// <summary>The entry for <paramref name="stack"/> (a stack with a byte element on top), or null when the cache is full.</summary>
    public TokenMaskEntry? Get(GrammarStack stack)
    {
        Span<int> positions = stackalloc int[Depth];
        int n = 0;
        var s = stack;
        while (s is not null && n < Depth)
        {
            positions[n++] = s.Position;
            s = s.Parent;
        }

        bool complete = s is null;
        string key = Key(positions[..n], complete);
        if (_entries.TryGetValue(key, out var entry))
        {
            return entry;
        }

        if (_entries.Count >= MaxEntries)
        {
            return null;
        }

        lock (_lock)
        {
            if (!_entries.TryGetValue(key, out entry))
            {
                entry = Compute(_runtime.Truncated(positions[..n], complete));
                _entries[key] = entry;
            }

            return entry;
        }
    }

    private static string Key(ReadOnlySpan<int> positions, bool complete)
    {
        var chars = new char[positions.Length * 2 + 1];
        for (int i = 0; i < positions.Length; i++)
        {
            chars[2 * i] = (char)(positions[i] >> 16);
            chars[2 * i + 1] = (char)positions[i];
        }

        chars[^1] = complete ? 'c' : 't';
        return new string(chars);
    }

    private TokenMaskEntry Compute(GrammarState start)
    {
        var tokens = new List<int>();
        var uncertain = new List<(int Start, int End)>();
        var childStart = _trie.ChildStart;
        var childByte = _trie.ChildByte;
        var childNode = _trie.ChildNode;
        var tokenStart = _trie.TokenStart;
        var all = _trie.Tokens;
        var dead = _runtime.Dead;
        _walk.Clear();
        _walk.Push((0, start, false));
        while (_walk.TryPop(out var top))
        {
            var state = top.State;
            for (int i = childStart[top.Node]; i < childStart[top.Node + 1]; i++)
            {
                byte b = childByte[i];
                if (!state.Allowed.Contains(b))
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
                    tokens.Add(all[t]);
                }

                bool covered = top.Covered;
                if (next.Exits && !covered && childStart[child] < childStart[child + 1])
                {
                    // Longer tokens continue after the known frames: their fate depends on the context.
                    uncertain.Add((tokenStart[child + 1], tokenStart[_trie.SubtreeEnd[child]]));
                    covered = true;
                }

                if (childStart[child] < childStart[child + 1] && !next.Allowed.IsEmpty)
                {
                    _walk.Push((child, next, covered));
                }
            }
        }

        if (tokens.Count < SparseLimit)
        {
            return new TokenMaskEntry(null, [.. tokens], [.. uncertain]);
        }

        var words = new uint[(_trie.VocabSize + 31) / 32];
        foreach (int t in tokens)
        {
            words[t >> 5] |= 1u << (t & 31);
        }

        return new TokenMaskEntry(words, null, [.. uncertain]);
    }
}
