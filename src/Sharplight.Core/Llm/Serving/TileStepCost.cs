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
            _decode[decodeRows] = _decode.TryGetValue(decodeRows, out double d) ? Smoothing * d + (1 - Smoothing) * ms : ms;
            return;
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

    /// <summary>Predicted milliseconds of a prompt-bearing step of <paramref name="rows"/> rows (0 before <see cref="Fitted"/>).</summary>
    public double PromptStepMs(int rows) => _fitted ? Math.Max(0, _a + _b * rows + _c * Tiles(rows)) : 0;

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
