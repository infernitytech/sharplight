namespace Sharplight.Llm.Serving;

/// <summary>
/// Online estimate of how a step's cost follows its token tiles, for scheduling prompt rows around tile boundaries: prompt-bearing
/// steps of at least one tile are fitted as <c>time ≈ a + b · rows + c · tiles</c> (<c>tiles = ⌈rows / tile⌉</c>, exponentially weighted
/// least squares), so <c>c</c> is what a step pays for starting another tile beyond its rows (large where the prompt-sized GEMMs run whole
/// token tiles, near 0 where the cost is linear in rows, such as expert GEMMs tiled per expert). Decode-only steps are kept per decode
/// row count, and a step that adds at most <see cref="HeldMaxRows"/> prompt rows to decode rows is priced against the decode-only step
/// of its decode rows: what a few prompt rows held back from a prompt step cost in a later step. Nothing here knows a model; every
/// price is measured on the steps the engine runs. Not thread-safe (the scheduler thread owns it).
/// </summary>
internal sealed class TileStepCost
{
    /// <summary>Weight of the previous statistics per new prompt step.</summary>
    private const double Decay = 0.97;
    /// <summary>Weight of the previous value of a per-row-count average.</summary>
    private const double Smoothing = 0.8;
    /// <summary>Prompt steps (spanning at least two tile counts) before the tile fit is used.</summary>
    private const int MinPromptSteps = 12;
    /// <summary>Held-row steps observed before their measured price replaces the fit's.</summary>
    private const int MinHeldSteps = 3;
    /// <summary>Share of a step's predicted time that a move between steps must save (<see cref="Moves"/>).</summary>
    private const double Margin = 0.02;
    /// <summary>Prompt-bearing steps with decode rows in (or next to) a token-tile bucket before its price per prompt row is used.</summary>
    private const int MinChunkSteps = 2;
    /// <summary>Weight of a token-tile bucket's previous prompt-row sums per new step in that bucket.</summary>
    private const double ChunkDecay = 0.9;
    /// <summary>Most a decode-only step counts in its row count's price, as a multiple of the price.</summary>
    private const double DecodeStallCap = 1.25;
    /// <summary>Share of the larger chunks' price per row that smaller chunks must save (<see cref="PrefersSmallChunks"/>).</summary>
    private const double MarginalMargin = 0.1;
    /// <summary>Ridge weight on the tile term per weighted step (pulls an unidentified tile cost towards 0).</summary>
    private const double TileRidge = 1e-3;

    private readonly double[] _s = new double[9]; // w, r, t, y, rr, rt, tt, ry, ty
    private readonly Dictionary<int, double> _decode = [];
    private double _a, _b, _c;
    private bool _fitted;
    private int _promptSteps;
    private int _minTiles = int.MaxValue, _maxTiles;
    private double _held;
    private int _heldSteps;
    private readonly Dictionary<long, (double ExtraMs, double Rows, int Steps)> _chunks = []; // per token tiles of prompt rows

    /// <param name="tileRows">Rows of one token tile (at least 2).</param>
    public TileStepCost(int tileRows)
    {
        TileRows = Math.Max(2, tileRows);
    }

    /// <summary>Rows of one token tile.</summary>
    public int TileRows { get; }

    /// <summary>Most prompt rows a step may hold back (or take beyond its budget) on these prices: a quarter of a tile.</summary>
    public int HeldMaxRows => TileRows / 4;

    /// <summary>Whether the tile fit has enough prompt steps of different tile counts.</summary>
    public bool Fitted => _fitted;

    /// <summary>Fitted milliseconds per started token tile (0 before <see cref="Fitted"/>).</summary>
    public double TileMs => _fitted ? _c : 0;

    /// <summary>Fitted milliseconds per row of a prompt-bearing step (0 before <see cref="Fitted"/>).</summary>
    public double RowMs => _fitted ? _b : 0;

    /// <summary>Adds a step: its decode (one-row) rows, its prompt rows and its time in milliseconds.</summary>
    public void Observe(int decodeRows, int promptRows, double ms)
    {
        if (ms <= 0 || decodeRows + promptRows <= 0)
        {
            return;
        }

        if (promptRows == 0)
        {
            // Faster steps pull the price down quickly (a first step of a new shape records its launches), slower ones count at most
            // DecodeStallCap times the price (a step that waited for the host's CPU share is no price of the device).
            _decode[decodeRows] = !_decode.TryGetValue(decodeRows, out double d) ? ms
                : ms < d ? 0.5 * (d + ms) : Smoothing * d + (1 - Smoothing) * Math.Min(ms, DecodeStallCap * d);
            return;
        }

        // A step faster than the decode-only price of its decode rows means a stale price, not free prompt rows: such steps are left out.
        if (decodeRows > 0 && DecodeOnlyMs(decodeRows) is { } decodeOnly && ms > decodeOnly * (1 - Margin))
        {
            ObserveExtra(decodeRows + promptRows, promptRows, Math.Max(0, ms - decodeOnly));
        }

        if (decodeRows > 0 && promptRows <= HeldMaxRows && _decode.TryGetValue(decodeRows, out double plain))
        {
            double extra = Math.Max(0, ms - plain);
            _held = _heldSteps == 0 ? extra : Smoothing * _held + (1 - Smoothing) * extra;
            _heldSteps++;
        }

        int rows = decodeRows + promptRows;
        if (rows < TileRows)
        {
            return;
        }

        double r = rows, t = Tiles(rows);
        Span<double> x = [1, r, t, ms, r * r, r * t, t * t, r * ms, t * ms];
        for (int i = 0; i < _s.Length; i++)
        {
            _s[i] = Decay * _s[i] + x[i];
        }

        _promptSteps++;
        _minTiles = Math.Min(_minTiles, (int)t);
        _maxTiles = Math.Max(_maxTiles, (int)t);
        Fit();
    }

    /// <summary>
    /// Predicted milliseconds a prompt step of <paramref name="rows"/> rows saves by carrying <paramref name="fewer"/> rows less (null
    /// before <see cref="Fitted"/>).
    /// </summary>
    public double? SavedMs(int rows, int fewer)
    {
        if (!_fitted || fewer <= 0 || fewer >= rows)
        {
            return null;
        }

        return Math.Max(0, _b * fewer + _c * (Tiles(rows) - Tiles(rows - fewer)));
    }

    /// <summary>
    /// Predicted milliseconds that <paramref name="rows"/> prompt rows (at most <see cref="HeldMaxRows"/>) add to a later step of decode
    /// rows: measured on such steps once a few were seen, else the fit's cost of the rows (null before <see cref="Fitted"/>).
    /// </summary>
    public double? HeldMs(int rows)
    {
        if (_heldSteps >= MinHeldSteps)
        {
            return _held;
        }

        return _fitted ? Math.Max(0, _b * rows) : null;
    }

    /// <summary>
    /// Whether moving <paramref name="rows"/> prompt rows out of a prompt step of <paramref name="stepRows"/> rows into a later step of
    /// decode rows saves time: the step's saving exceeds the rows' price in the later step by more than <see cref="Margin"/> of the step's
    /// predicted time (prices within the fit's noise decide nothing).
    /// </summary>
    public bool Moves(int stepRows, int rows)
    {
        return SavedMs(stepRows, rows) is { } saved && HeldMs(rows) is { } held && saved - held > Margin * PromptStepMs(stepRows);
    }

    /// <summary>
    /// Whether <paramref name="rows"/> prompt rows that would otherwise be left to a later step of decode rows cost less in a prompt step
    /// of <paramref name="stepRows"/> rows (them included), by more than <see cref="Margin"/> of that step's predicted time.
    /// </summary>
    public bool Takes(int stepRows, int rows)
    {
        return SavedMs(stepRows, rows) is { } added && HeldMs(rows) is { } held && held - added > Margin * PromptStepMs(stepRows);
    }

    /// <summary>
    /// Measured milliseconds per prompt row that the prompt rows of a step of <paramref name="stepRows"/> rows (decode rows included) add
    /// to its decode rows, over the decode-only step of the same decode rows: the decayed sums of the added time and of the prompt rows of
    /// the prompt-bearing decode steps with the same count of token tiles, or with fewer than <see cref="MinChunkSteps"/> of those also
    /// the neighbouring counts (null when too few such steps were seen). Keyed on the step's tiles, not its prompt rows: the prompt-sized
    /// GEMMs run whole token tiles of all rows. Measured, never extrapolated: prompt rows that ride the decode step's weight reads are cheap per row
    /// (dense models below the bandwidth / compute crossover), rows that make an expert GEMM read more experts are not.
    /// </summary>
    public double? MarginalRowMs(int stepRows)
    {
        if (stepRows <= 0)
        {
            return null;
        }

        long tiles = Tiles(stepRows);
        var (extra, rows, steps) = _chunks.GetValueOrDefault(tiles);
        if (steps < MinChunkSteps)
        {
            foreach (long near in (ReadOnlySpan<long>)[tiles - 1, tiles + 1])
            {
                var n = _chunks.GetValueOrDefault(near);
                extra += n.ExtraMs;
                rows += n.Rows;
                steps += n.Steps;
            }
        }

        return steps >= MinChunkSteps && rows > 0 ? Math.Max(0, extra) / rows : null;
    }

    /// <summary>
    /// Whether decoding steps of <paramref name="smallRows"/> rows (decode and prompt rows) cost less per prompt row than steps of
    /// <paramref name="largeRows"/>, by more than <see cref="MarginalMargin"/> of the larger steps' price (false until both were
    /// measured, <see cref="MarginalRowMs"/>).
    /// </summary>
    public bool PrefersSmallChunks(int smallRows, int largeRows)
    {
        return smallRows > 0 && largeRows > smallRows && MarginalRowMs(smallRows) is { } small && MarginalRowMs(largeRows) is { } large
               && large > 0 && small < (1 - MarginalMargin) * large;
    }

    /// <summary>Predicted milliseconds of a prompt-bearing step of <paramref name="rows"/> rows (0 before <see cref="Fitted"/>).</summary>
    public double PromptStepMs(int rows) => _fitted ? Math.Max(0, _a + _b * rows + _c * Tiles(rows)) : 0;

    /// <summary>
    /// The decode-only step price of <paramref name="decodeRows"/> rows: the lowest price measured for at least as many decode rows (a step
    /// with fewer rows costs no more, so a price inflated by a stall or a first run of a rarely seen row count is bounded by the well
    /// measured counts above it), else the price of the nearest count below (null before any decode-only step).
    /// </summary>
    private double? DecodeOnlyMs(int decodeRows)
    {
        double? atLeast = null;
        int below = 0;
        foreach (var (rows, ms) in _decode)
        {
            if (rows >= decodeRows)
            {
                atLeast = atLeast is { } m ? Math.Min(m, ms) : ms;
            }
            else if (rows > below)
            {
                below = rows;
            }
        }

        return atLeast ?? (below > 0 ? _decode[below] : null);
    }

    private void ObserveExtra(int stepRows, int promptRows, double extraMs)
    {
        long tiles = Tiles(stepRows);
        var (extra, rows, steps) = _chunks.GetValueOrDefault(tiles);
        _chunks[tiles] = (ChunkDecay * extra + extraMs, ChunkDecay * rows + promptRows, steps + 1);
    }

    private long Tiles(int rows) => (rows + TileRows - 1) / TileRows;

    /// <summary>Solves the 3 x 3 normal equations of the weighted fit (ridge on the tile term); keeps the last fit when singular.</summary>
    private void Fit()
    {
        if (_promptSteps < MinPromptSteps || _maxTiles == _minTiles)
        {
            return;
        }

        double w = _s[0], r = _s[1], t = _s[2], y = _s[3], rr = _s[4], rt = _s[5], tt = _s[6] + TileRidge * _s[0], ry = _s[7], ty = _s[8];
        // [w r t; r rr rt; t rt tt] [a b c]' = [y ry ty]'
        double det = w * (rr * tt - rt * rt) - r * (r * tt - rt * t) + t * (r * rt - rr * t);
        if (Math.Abs(det) < 1e-9 * Math.Max(1, w * rr * tt))
        {
            return;
        }

        double a = (y * (rr * tt - rt * rt) - r * (ry * tt - rt * ty) + t * (ry * rt - rr * ty)) / det;
        double b = (w * (ry * tt - ty * rt) - y * (r * tt - rt * t) + t * (r * ty - ry * t)) / det;
        double c = (w * (rr * ty - rt * ry) - r * (r * ty - ry * t) + y * (r * rt - rr * t)) / det;
        _a = a;
        _b = Math.Max(0, b);
        _c = Math.Max(0, c);
        _fitted = true;
    }
}
