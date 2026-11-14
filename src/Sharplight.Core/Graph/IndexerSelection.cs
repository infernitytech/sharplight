namespace Sharplight.Graph;

/// <summary>
/// The selection settings of an <see cref="StandardOps.IndexerTopK"/> node, read from its attributes (shared by the schema and the backends).
/// </summary>
/// <param name="Budget">Selected key positions per row (attribute <c>k</c>), a multiple of <paramref name="Group"/>.</param>
/// <param name="Group">Positions per candidate group (attribute <c>group</c>, default 1).</param>
/// <param name="Pooled">Groups are scored through their gated pooled key (attribute <c>group_score</c> = "pooled") instead of their best position.</param>
/// <param name="Local">The latest groups that are always picked (attribute <c>local</c>, default 0).</param>
/// <param name="Tail">The visible positions of the row's incomplete group follow the selection (attribute <c>tail</c>, pooled or mean groups only).</param>
/// <param name="PerHead">Every query head selects for itself without head weights (attribute <c>per_head</c>).</param>
/// <param name="Mean">Groups are scored through the mean of their keys (attribute <c>group_score</c> = "mean"), optionally RMS-normed and rotated.</param>
/// <param name="PoolNormEps">Epsilon of the RMS norm of a mean-pooled key (attribute <c>pool_norm_eps</c>; 0 = no norm, otherwise a norm weight input).</param>
/// <param name="PoolRopeDims">Leading channels of a mean-pooled key rotated (neox pairs) at its group's first position (attribute <c>pool_rope_dims</c>, 0 = none).</param>
/// <param name="PoolRopeBase">Rotary base of the pooled keys (attribute <c>pool_rope_base</c>).</param>
internal sealed record IndexerSelection(int Budget, int Group, bool Pooled, int Local, bool Tail, bool PerHead, bool Mean = false, float PoolNormEps = 0f,
    int PoolRopeDims = 0, float PoolRopeBase = 10000f)
{
    /// <summary>Output entries per selection row: the budget plus the tail's G − 1 entries.</summary>
    public int Width => Budget + (Tail ? Group - 1 : 0);

    /// <summary>Only complete groups are candidates (pooled and mean groups).</summary>
    public bool CompleteGroups => Pooled || Mean;

    /// <summary>Whether an input follows the head weights: the position gate (pooled) or the pool norm weight (mean with a norm).</summary>
    public bool HasPoolInput => Pooled || (Mean && PoolNormEps > 0f);

    /// <summary>Index of the first input after the fixed ones (q, key cache, [head weights], [position gate or pool norm]): the multi-sequence tables.</summary>
    public int TableInput => 2 + (PerHead ? 0 : 1) + (HasPoolInput ? 1 : 0);

    /// <summary>Reads and validates the attributes of an IndexerTopK node.</summary>
    /// <exception cref="ArgumentException">The attributes are inconsistent.</exception>
    public static IndexerSelection From(Attributes a)
    {
        long budget = a.Get("k", 0L);
        long group = a.Get("group", 1L);
        string score = a.Get("group_score", "max");
        var selection = new IndexerSelection((int)budget, (int)group, score == "pooled", (int)a.Get("local", 0L), a.Get("tail", false), a.Get("per_head", false),
            score == "mean", a.Get("pool_norm_eps", 0f), (int)a.Get("pool_rope_dims", 0L), a.Get("pool_rope_base", 10000f));
        if (budget < 1 || group < 1 || budget % group != 0)
        {
            throw new ArgumentException($"IndexerTopK needs k >= 1 as a multiple of group (k {budget}, group {group}).");
        }

        if (score is not ("max" or "pooled" or "mean") || selection.Local < 0 || (selection.Tail && !selection.CompleteGroups))
        {
            throw new ArgumentException($"IndexerTopK: group_score '{score}' (max, pooled, mean), local {selection.Local}, tail only with pooled or mean groups.");
        }

        if (selection.PoolRopeDims < 0 || selection.PoolRopeDims % 2 != 0)
        {
            throw new ArgumentException($"IndexerTopK: pool_rope_dims {selection.PoolRopeDims} must be even.");
        }

        return selection;
    }
}
