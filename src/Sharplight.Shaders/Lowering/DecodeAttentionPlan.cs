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
    /// Key ranges per row for a launch of <paramref name="rows"/> query rows; mirrors <c>decode_max_splits</c> (one range
    /// beyond <see cref="MaxRows"/>, <c>DEC_SPLIT_ROWS</c>).
    /// </summary>
    public long MaxSplits(long rows) => rows > MaxRows ? 1 : Math.Clamp(TargetWorkgroups / Math.Max(1, rows * Groups), 1, MaxSplitCount);
}
