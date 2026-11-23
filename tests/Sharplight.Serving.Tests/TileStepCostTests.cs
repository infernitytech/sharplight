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
}
