namespace Sharplight.Llm.Structured;

/// <summary>
/// One parse stack of the grammar's pushdown automaton, as an interned linked list: <see cref="Position"/> is the next
/// element to match, <see cref="Parent"/> the continuation once the current rule ends. Interning (per
/// <see cref="GrammarRuntime"/>) makes equal stacks the same object, so sets of stacks compare by reference.
/// </summary>
internal sealed class GrammarStack(int position, GrammarStack? parent, int id)
{
    /// <summary>Position in <see cref="Grammar.Elements"/>; -1 for the accepting (empty) stack.</summary>
    public int Position { get; } = position;
    public GrammarStack? Parent { get; } = parent;
    public int Id { get; } = id;
    /// <summary>Memoized closure: the stacks with a byte element on top that this stack stands for.</summary>
    public GrammarStack[]? Expanded { get; set; }
}
