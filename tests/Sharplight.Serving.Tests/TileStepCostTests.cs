using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>The tile-aware step prices of <see cref="TileStepCost"/>.</summary>
public sealed class TileStepCostTests
{
    /// <summary>Prompt steps whose cost follows whole tiles of 128 rows: 5 ms per tile, 0.002 ms per row, 10 ms per decode step.</summary>
    private static TileStepCost Tiled()
    {
        var cost = new TileStepCost(128);
        for (int i = 0; i < 40; i++)
        {
            cost.Observe(16, 0, 10);
            int prompt = 100 + 37 * i % 700;
            int rows = 16 + prompt;
            cost.Observe(16, prompt, 10 + 5 * ((rows + 127) / 128) + 0.002 * rows);
        }

        return cost;
    }

    [Fact]
    public void FitsTheTileCostOfTiledSteps()
    {
        var cost = Tiled();
        Assert.True(cost.Fitted);
        Assert.InRange(cost.TileMs, 4.9, 5.1);
        Assert.InRange(cost.RowMs, 0.0015, 0.0025);

        // 266 rows: holding 10 back drops the third tile; 250 rows: holding 10 drops nothing but the rows.
        Assert.InRange(cost.SavedMs(266, 10)!.Value, 4.9, 5.1);
        Assert.InRange(cost.SavedMs(250, 10)!.Value, 0, 0.03);
    }

    [Fact]
    public void LinearStepsHaveNoTileCost()
    {
        var cost = new TileStepCost(128);
        for (int i = 0; i < 40; i++)
        {
            int prompt = 100 + 37 * i % 700;
            cost.Observe(16, prompt, 12 + 0.05 * (16 + prompt));
        }

        Assert.True(cost.Fitted);
        Assert.InRange(cost.TileMs, 0, 0.05);
        Assert.InRange(cost.SavedMs(266, 10)!.Value, 0.45, 0.55);
    }

    [Fact]
    public void HeldRowsArePricedOnDecodeStepsThatCarryThem()
    {
        var cost = Tiled();
        // Before such steps were seen, the fit's row price applies.
        Assert.InRange(cost.HeldMs(10)!.Value, 0, 0.03);
        for (int i = 0; i < 5; i++)
        {
            cost.Observe(16, 10, 12.5);
        }

        Assert.InRange(cost.HeldMs(10)!.Value, 2.4, 2.6);
        Assert.Equal(32, cost.HeldMaxRows);
    }

    [Fact]
    public void NoPricesBeforeEnoughSteps()
    {
        var cost = new TileStepCost(128);
        cost.Observe(16, 300, 30);
        Assert.False(cost.Fitted);
        Assert.Null(cost.SavedMs(266, 10));
        Assert.Null(cost.HeldMs(10));
    }

    /// <summary>
    /// Steps of 16 decode rows (12 ms alone) carrying <paramref name="rowMs"/>(prompt rows) per prompt row, as measured on a dense FP8 model
    /// (rising: prompt rows ride the decode step's weight reads until the GEMMs turn compute-bound) or a mixture of experts (falling).
    /// </summary>
    private static TileStepCost Marginal(Func<int, double> rowMs)
    {
        var cost = new TileStepCost(128);
        int[] prompts = [48, 112, 176, 240, 368, 496, 752, 240, 496, 112];
        foreach (int p in prompts)
        {
            cost.Observe(16, 0, 12);
            cost.Observe(16, p, 12 + p * rowMs(p));
        }

        return cost;
    }

    [Fact]
    public void RisingRowPricesPreferSmallChunks()
    {
        // 0.015 ms per row at 48 rows to 0.052 at 752 (Qwen3-8B-FP8, 16 decode rows; benchmarks.md §R47).
        var cost = Marginal(p => 0.013 + 0.00005 * p);
        Assert.InRange(cost.MarginalRowMs(240)!.Value, 0.024, 0.026);
        Assert.True(cost.PrefersSmallChunks(240, 752));
    }

    [Fact]
    public void FallingOrFlatRowPricesKeepLargeChunks()
    {
        Assert.False(Marginal(p => 0.065 - 0.00002 * p).PrefersSmallChunks(240, 752));
        Assert.False(Marginal(_ => 0.05).PrefersSmallChunks(240, 752));
        // A fixed price per prompt-bearing step (1.5 ms) and 0.05 ms per row, as on a mixture of experts: large chunks are cheaper per row.
        Assert.False(Marginal(p => 1.5 / p + 0.05).PrefersSmallChunks(240, 752));
    }

    [Fact]
    public void NoChunkPreferenceBeforeEnoughSteps()
    {
        var cost = new TileStepCost(128);
        cost.Observe(16, 0, 12);
        cost.Observe(16, 100, 14);
        cost.Observe(16, 700, 50);
        Assert.Null(cost.MarginalRowMs(240));
        Assert.False(cost.PrefersSmallChunks(240, 752));
    }
}
