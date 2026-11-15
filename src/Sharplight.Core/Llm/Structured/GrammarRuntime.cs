namespace Sharplight.Llm.Structured;

/// <summary>
/// Executes a <see cref="Grammar"/> byte by byte as a pushdown automaton over sets of interned stacks (the llama.cpp
/// approach), memoizing stack closures and state transitions. One runtime per matcher; not thread-safe.
/// </summary>
internal sealed class GrammarRuntime
{
    /// <summary>Guards against grammars whose expansion would not terminate (rejected at compile time, kept as a backstop).</summary>
    private const int MaxDepth = 10_000;

    private readonly Grammar _grammar;
    private readonly Dictionary<(int Position, int Parent), GrammarStack> _stacks = [];
    private readonly Dictionary<StateKey, GrammarState> _states = [];
    private readonly GrammarStack _accept;
    /// <summary>Bottom of a truncated stack (<see cref="Truncated"/>): reaching it means the input left the known frames.</summary>
    private readonly GrammarStack _exit;
    private readonly List<GrammarStack> _scratch = [];
    private readonly HashSet<GrammarStack> _seen = new(ReferenceEqualityComparer.Instance);

    public GrammarRuntime(Grammar grammar)
    {
        _grammar = grammar;
        _accept = new GrammarStack(-1, null, 0);
        _exit = new GrammarStack(-2, null, -2);
        _exit.Expanded = [_exit];
        Dead = new GrammarState(-1, [], default, false, false);
        _scratch.Clear();
        foreach (int start in grammar.Alternatives[0])
        {
            AddUnique(Expand(Intern(start, null), 0));
        }

        Initial = InternState();
    }

    /// <summary>The state before any input.</summary>
    public GrammarState Initial { get; }
    /// <summary>The state after input no match can start with.</summary>
    public GrammarState Dead { get; }
    /// <summary>Number of distinct states created so far (bounded by the inputs seen).</summary>
    public int StateCount => _states.Count;

    /// <summary>The state after <paramref name="b"/>; <see cref="Dead"/> when no stack accepts it.</summary>
    public GrammarState Advance(GrammarState state, byte b)
    {
        if (state.GetNext(b) is { } cached)
        {
            return cached;
        }

        GrammarState next = Dead;
        if (state.Allowed.Contains(b))
        {
            _scratch.Clear();
            _seen.Clear();
            foreach (var stack in state.Stacks)
            {
                if (stack.Position >= 0 && _grammar.Elements[stack.Position].Set.Contains(b))
                {
                    AddUnique(Expand(Intern(stack.Position + 1, stack.Parent), 0));
                }
            }

            next = _scratch.Count == 0 ? Dead : InternState();
        }

        state.SetNext(b, next);
        return next;
    }

    /// <summary>The state after all of <paramref name="bytes"/>.</summary>
    public GrammarState Advance(GrammarState state, ReadOnlySpan<byte> bytes)
    {
        foreach (byte b in bytes)
        {
            if (state == Dead)
            {
                break;
            }

            state = Advance(state, b);
        }

        return state;
    }

    /// <summary>
    /// The state of a single stack given by its top <paramref name="positions"/> (top first, each a position after a rule
    /// reference except the first, a byte element). Below them is either the end of the grammar
    /// (<paramref name="complete"/>) or the exit marker, which <see cref="GrammarState.Exits"/> reports once reached.
    /// </summary>
    public GrammarState Truncated(ReadOnlySpan<int> positions, bool complete)
    {
        GrammarStack? stack = complete ? null : _exit;
        for (int i = positions.Length - 1; i >= 0; i--)
        {
            stack = Intern(positions[i], stack);
        }

        _scratch.Clear();
        _seen.Clear();
        AddUnique(Expand(stack!, 0));
        return InternState();
    }

    private void AddUnique(GrammarStack[] stacks)
    {
        foreach (var s in stacks)
        {
            if (_seen.Add(s))
            {
                _scratch.Add(s);
            }
        }
    }

    private GrammarStack Intern(int position, GrammarStack? parent)
    {
        var key = (position, parent?.Id ?? -1);
        if (!_stacks.TryGetValue(key, out var stack))
        {
            stack = new GrammarStack(position, parent, _stacks.Count + 1);
            _stacks.Add(key, stack);
        }

        return stack;
    }

    /// <summary>The stacks with a byte element on top (or the accepting stack) that <paramref name="stack"/> stands for.</summary>
    private GrammarStack[] Expand(GrammarStack stack, int depth)
    {
        if (stack.Expanded is { } done)
        {
            return done;
        }

        if (depth > MaxDepth)
        {
            throw new InvalidOperationException("The grammar expands without consuming input (left recursion).");
        }

        var element = _grammar.Elements[stack.Position];
        GrammarStack[] result;
        switch (element.Kind)
        {
            case Grammar.ElementKind.Bytes:
                result = [stack];
                break;
            case Grammar.ElementKind.End:
                result = stack.Parent is null ? [_accept] : Expand(stack.Parent, depth + 1);   // the exit stack expands to itself
                break;
            default:
            {
                // Tail call: a rule referenced last in its alternative continues directly with the parent, so right
                // recursion (repetitions) keeps the stack flat and its states recur instead of growing with the input.
                var continuation = _grammar.Elements[stack.Position + 1].Kind == Grammar.ElementKind.End
                    ? stack.Parent
                    : Intern(stack.Position + 1, stack.Parent);
                var parts = new List<GrammarStack>();
                var seen = new HashSet<GrammarStack>(ReferenceEqualityComparer.Instance);
                foreach (int start in _grammar.Alternatives[element.Rule])
                {
                    foreach (var s in Expand(Intern(start, continuation), depth + 1))
                    {
                        if (seen.Add(s))
                        {
                            parts.Add(s);
                        }
                    }
                }

                result = [.. parts];
                break;
            }
        }

        stack.Expanded = result;
        return result;
    }

    /// <summary>Interns the stack set collected in <see cref="_scratch"/>.</summary>
    private GrammarState InternState()
    {
        _scratch.Sort(static (a, b) => a.Id.CompareTo(b.Id));
        var ids = new int[_scratch.Count];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = _scratch[i].Id;
        }

        var key = new StateKey(ids);
        if (_states.TryGetValue(key, out var state))
        {
            return state;
        }

        ByteSet allowed = default;
        bool accepting = false;
        bool exits = false;
        foreach (var s in _scratch)
        {
            if (s.Position == -1)
            {
                accepting = true;
            }
            else if (s.Position == -2)
            {
                exits = true;
            }
            else
            {
                allowed = allowed.Union(_grammar.Elements[s.Position].Set);
            }
        }

        state = new GrammarState(_states.Count, [.. _scratch], allowed, accepting, exits);
        _states.Add(key, state);
        return state;
    }

    /// <summary>Structural key of a stack set (sorted stack ids).</summary>
    private readonly struct StateKey(int[] ids) : IEquatable<StateKey>
    {
        private readonly int _hash = Hash(ids);

        public bool Equals(StateKey other) => _hash == other._hash && ids.AsSpan().SequenceEqual(other.Ids);

        public override bool Equals(object? obj) => obj is StateKey k && Equals(k);

        public override int GetHashCode() => _hash;

        private int[] Ids => ids;

        private static int Hash(int[] ids)
        {
            var h = new HashCode();
            foreach (int id in ids)
            {
                h.Add(id);
            }

            return h.ToHashCode();
        }
    }
}
