namespace Sharplight.Architectures.SeedVr;

/// <summary>
/// The index tables that run window attention with generic ops: every window attends over its own video tokens followed by all text tokens.
/// The windows are grouped by size; per group the query, key and value rows of its windows are gathered into one packed batch (a window's
/// tokens in (frame, row, column) order, then the text rows) from the rows [video tokens..., text tokens...], so one batched attention per
/// group covers all its windows. The groups' outputs, concatenated, are gathered back into token order, and the text rows of every window
/// are averaged.
/// </summary>
internal sealed class WindowAttentionPlan
{
    private WindowAttentionPlan(IReadOnlyList<VideoWindow> windows, IReadOnlyList<WindowGroup> groups, int[] tokenRows, int[] textRows)
    {
        Windows = windows;
        Groups = groups;
        TokenRows = tokenRows;
        TextRows = textRows;
    }

    /// <summary>The windows, in the official order.</summary>
    public IReadOnlyList<VideoWindow> Windows { get; }

    /// <summary>The groups of equally sized windows.</summary>
    public IReadOnlyList<WindowGroup> Groups { get; }

    /// <summary>For every video token, its row in the concatenated group outputs.</summary>
    public int[] TokenRows { get; }

    /// <summary>For every window and text token (window-major), its row in the concatenated group outputs.</summary>
    public int[] TextRows { get; }

    /// <summary>Plans the windows of a <paramref name="frames"/> × <paramref name="rows"/> × <paramref name="columns"/> grid.</summary>
    /// <param name="windows">The windows (they tile the grid).</param>
    /// <param name="frames">Frames of the grid.</param>
    /// <param name="rows">Rows of the grid.</param>
    /// <param name="columns">Columns of the grid.</param>
    /// <param name="textRows">Text tokens appended to every window.</param>
    public static WindowAttentionPlan Create(IReadOnlyList<VideoWindow> windows, int frames, int rows, int columns, int textRows)
    {
        int tokens = frames * rows * columns;
        var tokenRows = new int[tokens];
        var textRowIndex = new int[windows.Count * textRows];
        var groups = new List<WindowGroup>();
        int outputRow = 0;
        var bySize = windows.Select((window, index) => (window, index)).GroupBy(w => (w.window.Frames, w.window.Rows, w.window.Columns));
        foreach (var group in bySize)
        {
            var members = group.ToList();
            int size = members[0].window.Size;
            int sequence = size + textRows;
            var gather = new int[members.Count * sequence];
            int at = 0;
            foreach (var (window, index) in members)
            {
                for (int f = window.FrameStart; f < window.FrameEnd; f++)
                {
                    for (int r = window.RowStart; r < window.RowEnd; r++)
                    {
                        for (int c = window.ColumnStart; c < window.ColumnEnd; c++)
                        {
                            int token = (f * rows + r) * columns + c;
                            gather[at] = token;
                            tokenRows[token] = outputRow + at;
                            at++;
                        }
                    }
                }

                for (int t = 0; t < textRows; t++)
                {
                    gather[at] = tokens + t;
                    textRowIndex[index * textRows + t] = outputRow + at;
                    at++;
                }
            }

            groups.Add(new WindowGroup(members.Count, sequence, gather));
            outputRow += gather.Length;
        }

        return new WindowAttentionPlan(windows, groups, tokenRows, textRowIndex);
    }
}
