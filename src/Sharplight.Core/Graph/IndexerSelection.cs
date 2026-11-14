namespace Sharplight.Graph;

/// <summary>
/// The selection settings of an <see cref="StandardOps.IndexerTopK"/> node, read from its attributes (shared by the schema and the backends).
/// </summary>
/// <param name="Budget">Selected key positions per row (attribute <c>k</c>), a multiple of <paramref name="Group"/>.</param>
/// <param name="Group">Positions per candidate group (attribute <c>group</c>, default 1).</param>
/// <param name="Pooled">Groups are scored through their gated pooled key (attribute <c>group_score</c> = "pooled") instead of their best position.</param>
/// <param name="Local">The latest groups that are always picked (attribute <c>local</c>, default 0).</param>
/// <param name="Tail">The visible positions of the row's incomplete group follow the selection (attribute <c>tail</c>, pooled groups only).</param>
/// <param name="PerHead">Every query head selects for itself without head weights (attribute <c>per_head</c>).</param>
internal sealed record IndexerSelection(int Budget, int Group, bool Pooled, int Local, bool Tail, bool PerHead)
{
    /// <summary>Output entries per selection row: the budget plus the tail's G − 1 entries.</summary>
    public int Width => Budget + (Tail ? Group - 1 : 0);

    /// <summary>Index of the first input after the fixed ones (q, key cache, [head weights], [position gate]): the multi-sequence tables.</summary>
    public int TableInput => 2 + (PerHead ? 0 : 1) + (Pooled ? 1 : 0);

    /// <summary>Reads and validates the attributes of an IndexerTopK node.</summary>
    /// <exception cref="ArgumentException">The attributes are inconsistent.</exception>
    public static IndexerSelection From(Attributes a)
    {
        long budget = a.Get("k", 0L);
        long group = a.Get("group", 1L);
        string score = a.Get("group_score", "max");
        var selection = new IndexerSelection((int)budget, (int)group, score == "pooled", (int)a.Get("local", 0L), a.Get("tail", false), a.Get("per_head", false));
        if (budget < 1 || group < 1 || budget % group != 0)
        {
            throw new ArgumentException($"IndexerTopK needs k >= 1 as a multiple of group (k {budget}, group {group}).");
        }

        if (score is not ("max" or "pooled") || selection.Local < 0 || (selection.Tail && !selection.Pooled))
        {
            throw new ArgumentException($"IndexerTopK: group_score '{score}' (max, pooled), local {selection.Local}, tail only with pooled groups.");
        }

        return selection;
    }
}
