namespace Sharplight.Shaders.Lowering;

/// <summary>Configuration of split-K flash decoding for one attention node (see <c>decode</c> in attention.slang).</summary>
internal sealed class DecodeAttentionPlan
{
    /// <summary>Kernel macros of <c>decode</c> and <c>decode_combine</c>.</summary>
    public required Dictionary<string, string> Macros { get; init; }

    /// <summary>Workgroups per (query row, key range): KV heads x query-head groups per KV head.</summary>
    public required long Groups { get; init; }

    /// <summary>Query rows up to which the kernel is used.</summary>
    public required long MaxRows { get; init; }

    /// <summary>Workgroups the split count aims for (<c>DEC_TARGET_WG</c>).</summary>
    public required long TargetWorkgroups { get; init; }

    /// <summary>Upper bound of the split count (<c>DEC_MAX_SPLITS</c>).</summary>
    public required long MaxSplitCount { get; init; }

    /// <summary>Bindings for which the decode kernel (and its combine) run.</summary>
    public required Func<Bindings, bool> When { get; init; }

    /// <summary>
    /// Launches of up to this many query rows split each row's keys as a single-row launch does (<c>DEC_INVARIANT_ROWS</c>),
    /// so every row's result is the same as in a launch of its own.
    /// </summary>
    public long InvariantRows { get; init; }

    /// <summary>
    /// True when no row ever has more keys than one range takes (a sliding window of at most <c>DEC_MIN_KEYS</c> keys): every row
    /// then runs one key range whatever the split count, so the launch needs no further ranges and no combine.
    /// </summary>
    public bool SingleRange { get; init; }

    /// <summary>
    /// Key ranges per row for a launch of <paramref name="rows"/> query rows; mirrors <c>decode_max_splits</c> (one range
    /// beyond <see cref="MaxRows"/>, <c>DEC_SPLIT_ROWS</c>; the single-row count up to <see cref="InvariantRows"/>), and one for a
    /// <see cref="SingleRange"/> plan.
    /// </summary>
    public long MaxSplits(long rows) => rows > MaxRows || SingleRange ? 1
        : Math.Clamp(TargetWorkgroups / Math.Max(1, (rows <= InvariantRows ? 1 : rows) * Groups), 1, MaxSplitCount);

    /// <summary>
    /// True when a launch of <paramref name="rows"/> query rows may split a row's keys and so needs the combine (one range per row
    /// writes the output directly).
    /// </summary>
    public bool Combines(long rows) => MaxSplits(rows) > 1;
}
