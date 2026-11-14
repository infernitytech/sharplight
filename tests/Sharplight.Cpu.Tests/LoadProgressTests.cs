using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu.Tests;

public class LoadProgressTests
{
    private static ComputeGraph SmallGraph()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 16);
        var h = tg.B.MatMul(x, tg.WF("w1", Kit.Rand(16 * 16, 1), 16, 16));
        h = tg.B.Unary("Silu", h);
        h = tg.B.MatMul(h, tg.WF("w2", Kit.Rand(8 * 16, 2), 8, 16));
        tg.B.Output("y", tg.B.Unary("Tanh", h));
        return tg.B.Build();
    }

    private static CompiledModel Compile(ComputeGraph graph, IProgress<LoadProgress>? progress, CancellationToken ct = default)
        => CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = Kit.Device }],
            ReservePerDevice = 0,
            Compile = new CompileOptions { MaxTokens = 4, ContextLength = 4, Progress = progress, CancellationToken = ct },
        });

    [Fact]
    public void CpuCompileReportsPlacingThenCompilingUpToTotal()
    {
        var log = new ProgressLog();
        using var m = Compile(SmallGraph(), log);

        Assert.Equal([LoadStage.Placing, LoadStage.Compiling], log.Stages);
        Assert.True(log.Last(LoadStage.Placing).IsComplete);
        var compiling = log.Reports.Where(r => r.Stage == LoadStage.Compiling).ToList();
        long nodes = m.Plan.Segments.Sum(s => s.Nodes.Count);
        Assert.Equal(0, compiling[0].Completed);
        Assert.Equal(nodes, compiling[0].Total);
        Assert.Equal(nodes, compiling[^1].Completed);
        Assert.Equal(nodes, compiling[^1].Total);
        Assert.True(compiling.Zip(compiling.Skip(1)).All(p => p.First.Completed < p.Second.Completed), "reports must grow");
    }

    [Fact]
    public void CancelledCompileThrows()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => Compile(SmallGraph(), null, cts.Token));
    }

    [Fact]
    public void CancellationDuringCompileStopsIt()
    {
        using var cts = new CancellationTokenSource();
        var log = new CancelOnReport(LoadStage.Compiling, cts);
        Assert.ThrowsAny<OperationCanceledException>(() => Compile(SmallGraph(), log, cts.Token));
    }

    [Fact]
    public void CounterReportsAtBoundedRateAndAlwaysTheTotal()
    {
        var log = new ProgressLog();
        var counter = new LoadProgressCounter(log, LoadStage.Uploading, 10_000, "dev", cached: 3, maxReports: 50);
        Parallel.For(0, 10_000, _ => counter.Add());
        counter.Complete();

        var reports = log.Reports;
        Assert.InRange(reports.Count, 3, 52);
        Assert.Equal(0, reports[0].Completed);
        Assert.Equal(10_000, reports[^1].Completed);
        Assert.All(reports, r => Assert.Equal(10_000, r.Total));
        Assert.All(reports, r => Assert.Equal(3, r.Cached));
        Assert.True(reports.Zip(reports.Skip(1)).All(p => p.First.Completed < p.Second.Completed), "reports must grow");
        Assert.Equal(10_000, counter.Completed);
    }

    [Fact]
    public void CounterWithoutWorkReportsDoneOnce()
    {
        var log = new ProgressLog();
        var counter = new LoadProgressCounter(log, LoadStage.Compiling, 0);
        counter.Complete();
        var only = Assert.Single(log.Reports);
        Assert.True(only.IsComplete);
        Assert.Equal(1, only.Fraction);
    }
}
