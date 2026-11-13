namespace Sharplight.Llm.Structured;

/// <summary>
/// An interned set of parse stacks: the state of the grammar after some prefix. Transitions per byte and the allowed-token
/// mask are memoized on the state, so states that recur (inside a JSON string, between array items) cost one lookup.
/// </summary>
internal sealed class GrammarState(int id, GrammarStack[] stacks, ByteSet allowed, bool accepting, bool exits)
{
    private GrammarState?[]? _next;

    public int Id { get; } = id;
    /// <summary>The stacks, sorted by id; each has a byte element on top or is the accepting stack.</summary>
    public GrammarStack[] Stacks { get; } = stacks;
    /// <summary>Bytes that some stack can consume next.</summary>
    public ByteSet Allowed { get; } = allowed;
    /// <summary>True when the input so far is a complete match.</summary>
    public bool Accepting { get; } = accepting;
    /// <summary>True when the input left the frames of a truncated stack (see <see cref="GrammarRuntime.Truncated"/>).</summary>
    public bool Exits { get; } = exits;
    /// <summary>Allowed-token mask words for this state (see <see cref="GrammarMatcher"/>), when cached.</summary>
    public uint[]? CachedMask { get; set; }

    public GrammarState? GetNext(byte b) => _next?[b];

    public void SetNext(byte b, GrammarState next) => (_next ??= new GrammarState?[256])[b] = next;
}
