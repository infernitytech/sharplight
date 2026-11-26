using Sharplight.Models;
using Sharplight.Studio.Controls;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.TimeSeries;
using Sharplight.TimeSeries;

namespace Sharplight.Studio.Tests;

/// <summary>The Time series page over a fake forecaster: reading the table, choosing the column, the run, the chart series, the export and the interval levels.</summary>
public sealed class TimeSeriesPageTests
{
    private const string Daily = "date,sales,visits\n2024-01-01,10,100\n2024-01-02,11,101\n2024-01-03,12,103\n2024-01-04,13,104\n2024-01-05,14,106\n";

    private static string WriteCsv(TestServices services, string text, string name = "series.csv")
    {
        string path = services.Directory + "/" + name;
        File.WriteAllText(path, text);
        return path;
    }

    private static FakeModel Forecaster(TestServices services, out FakeForecaster client)
    {
        client = new FakeForecaster();
        var model = new FakeModel("chronos-bolt-tiny") { Kind = ModelKinds.TimeSeriesForecasting };
        model.Clients[typeof(ITimeSeriesForecaster)] = client;
        services.Runtime.Add(model);
        return model;
    }

    private static void Set(TimeSeriesPageViewModel page, string key, string value)
        => page.Panel.Parameters.Single(parameter => parameter.Key == key).Deserialize(value);

    [Fact]
    public void Interval_MapsCoverageToLevels()
    {
        Assert.Equal((0.1f, 0.9f), ForecastInterval.Levels(80));
        Assert.Equal((0.25f, 0.75f), ForecastInterval.Levels(50));
        Assert.Equal((0.05f, 0.95f), ForecastInterval.Levels(90));
        Assert.Equal([50, 80, 90], ForecastInterval.Coverages);
    }

    [Fact]
    public void Timeline_ContinuesConstantSteps_AndRefusesIrregularOnes()
    {
        Assert.Equal(["2024-01-06", "2024-01-07"], ForecastTimeline.Continue(["2024-01-04", "2024-01-05"], 2));
        Assert.Equal(["2024-03-01 00:30:00"], ForecastTimeline.Continue(["2024-03-01 00:00:00", "2024-03-01 00:15:00"], 1));
        Assert.Equal(["2024-04-01", "2024-05-01"], ForecastTimeline.Continue(["2024-01-01", "2024-02-01", "2024-03-01"], 2));
        Assert.Null(ForecastTimeline.Continue(["2024-01-01", "2024-01-02", "2024-01-05"], 2));
        Assert.Null(ForecastTimeline.Continue(["a", "b"], 2));
        Assert.Null(ForecastTimeline.Continue(null, 2));
    }

    [Fact]
    public async Task Table_ListsTheNumericColumns_AndChoosesTheFirst()
    {
        using var services = new TestServices();
        var page = new TimeSeriesPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();

        Assert.False(page.HasTable);
        Assert.Contains("Load a time series model", page.RunHint, StringComparison.Ordinal);
        Assert.True(await page.SetTableAsync(WriteCsv(services, Daily)));

        Assert.Equal(["sales", "visits"], page.ColumnNames);
        Assert.Equal("sales", page.SelectedColumn);
        Assert.Equal("series.csv", page.TableName);
        Assert.Equal("5 rows · 2 numeric columns · time column date", page.TableSummary);
        Assert.False(page.RunCommand.CanExecute(null));
    }

    [Fact]
    public async Task Table_RefusesBadFiles_WithANotice()
    {
        using var services = new TestServices();
        var page = new TimeSeriesPageViewModel(services.Services, new FakeNavigator());

        Assert.False(await page.SetTableAsync(services.Directory + "/missing.csv"));
        Assert.Contains("does not exist", page.Notice, StringComparison.Ordinal);
        Assert.False(await page.SetTableAsync(WriteCsv(services, "name,city\nada,paris\nbob,rome\n", "words.csv")));
        Assert.Contains("words.csv cannot be read", page.Notice, StringComparison.Ordinal);
        Assert.False(page.HasTable);
        Assert.False(await page.SetTableAsync(WriteCsv(services, "", "empty.csv")));
        Assert.True(page.HasNotice);
    }

    [Fact]
    public async Task Run_ForecastsTheChosenColumn_AndShowsHistoryForecastAndBand()
    {
        using var services = new TestServices();
        Forecaster(services, out var client);
        var page = new TimeSeriesPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.ModelProbe;
        Assert.Equal("context 512 · prediction length 64", page.ModelInfo);
        Assert.Equal(64, page.Panel.Options.Horizon);

        await page.SetTableAsync(WriteCsv(services, Daily));
        page.SelectedColumn = "visits";
        Set(page, "horizon", "3");
        Set(page, "interval", "90");
        Set(page, "samples", "7");
        Set(page, "seed", "42");
        Assert.True(page.RunCommand.CanExecute(null));
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(RunStatus.Completed, page.Queue.Items[0].Status);
        Assert.Equal([100f, 101f, 103f, 104f, 106f], client.Histories.Single());
        var options = Assert.Single(client.Requests)!;
        Assert.Equal(3, options.Horizon);
        Assert.Equal([0.05f, 0.5f, 0.95f], options.QuantileLevels);
        Assert.Equal(7, options.SampleCount);
        Assert.Equal(42, options.Seed);

        // History (the last 16 points at least, here the 5 there are), then 3 forecast steps continuing it, with the band at the 5 % and 95 % levels.
        Assert.True(page.HasResult);
        Assert.Equal(5, page.HistoryPoints.Count);
        Assert.Equal(3, page.ForecastPoints.Count);
        Assert.Equal(3, page.LowerPoints.Count);
        Assert.Equal(3, page.UpperPoints.Count);
        Assert.Equal(new ChartPoint(5, 107), page.ForecastPoints[0]);
        Assert.Equal(new ChartPoint(7, 109), page.ForecastPoints[2]);
        Assert.Equal(102.5, page.LowerPoints[0].Y, 3);
        Assert.Equal(113.5, page.UpperPoints[2].Y, 3);
        Assert.Equal("2024-01-01", page.ChartStart);
        Assert.Equal("2024-01-08", page.ChartEnd);
        Assert.Equal(["5 %", "95 %"], new[] { page.LowHeader, page.HighHeader });
        Assert.Equal(3, page.Rows.Count);
        Assert.Equal("2024-01-06", page.Rows[0].Time);
        Assert.Contains("3 steps", page.ResultSummary, StringComparison.Ordinal);
        Assert.Contains("prediction length 64", page.ResultSummary, StringComparison.Ordinal);
        Assert.Single(page.History.Items);
        Assert.Equal(["forecast.csv", "forecast.json"], Assert.Single(page.Gallery.Items).Files.Select(file => file.Name));
    }

    [Fact]
    public async Task Run_PlotsAtMostFourHorizonsOfHistory_AndTheStepIndexWithoutATimeColumn()
    {
        using var services = new TestServices();
        Forecaster(services, out var client);
        var page = new TimeSeriesPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.SetTableAsync(WriteCsv(services, "value\n" + string.Join('\n', Enumerable.Range(1, 100)) + "\n"));
        Set(page, "horizon", "5");
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(100, client.Histories.Single().Length);
        Assert.Equal(20, page.HistoryPoints.Count);
        Assert.Equal(5, page.ForecastPoints.Count);
        Assert.Equal(new ChartPoint(20, 101), page.ForecastPoints[0]);
        Assert.Equal("-20", page.ChartStart);
        Assert.Equal("+5", page.ChartEnd);
        Assert.False(page.HasTimes);
    }

    [Fact]
    public async Task Export_WritesStepsMeanAndEveryQuantile()
    {
        using var services = new TestServices();
        Forecaster(services, out _);
        var page = new TimeSeriesPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.SetTableAsync(WriteCsv(services, "value\n1\n2\n3\n4\n", "plain.csv"));
        Set(page, "horizon", "2");
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        string target = services.Directory + "/out.csv";
        services.Platform.SaveTo = target;
        await page.ExportCsvCommand.ExecuteAsync(null);
        string text = await File.ReadAllTextAsync(target, TestContext.Current.CancellationToken);
        Assert.Equal("step,mean,q0.1,q0.5,q0.9\n1,5,1,5,9\n2,6,2,6,10\n", text.Replace("\r\n", "\n", StringComparison.Ordinal));
        Assert.Equal(text, await File.ReadAllTextAsync(page.Gallery.Items[0].Files[0].Path, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Export_UsesTheContinuedTimestamps_WhenTheTimeColumnHasAConstantStep()
    {
        using var services = new TestServices();
        Forecaster(services, out _);
        var page = new TimeSeriesPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.SetTableAsync(WriteCsv(services, Daily));
        Set(page, "horizon", "2");
        Set(page, "interval", "50");
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        string target = services.Directory + "/dated.csv";
        services.Platform.SaveTo = target;
        await page.ExportCsvCommand.ExecuteAsync(null);
        string[] lines = (await File.ReadAllTextAsync(target, TestContext.Current.CancellationToken)).Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal("date,mean,q0.25,q0.5,q0.75", lines[0]);
        Assert.Equal("2024-01-06,15,12.5,15,17.5", lines[1]);
        Assert.Equal("2024-01-07,16,13.5,16,18.5", lines[2]);
        Assert.True(page.HasTimes);
    }

    [Fact]
    public async Task Run_ShowsAFailureInTheNotice_AndKeepsItInTheHistory()
    {
        using var services = new TestServices();
        Forecaster(services, out var client);
        client.Failure = new ArgumentException("The history has no observed value.");
        var page = new TimeSeriesPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.SetTableAsync(WriteCsv(services, Daily));
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(RunStatus.Failed, page.Queue.Items[0].Status);
        Assert.Equal("The history has no observed value.", page.Notice);
        Assert.False(page.HasResult);
        Assert.Equal("The history has no observed value.", page.History.Items.Single().Record.Error);
    }

    [Fact]
    public async Task History_RestoresTheForecastAndTheParameters()
    {
        using var services = new TestServices();
        Forecaster(services, out _);
        var page = new TimeSeriesPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.SetTableAsync(WriteCsv(services, Daily));
        Set(page, "horizon", "4");
        Set(page, "interval", "50");
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        var reopened = new TimeSeriesPageViewModel(services.Services, new FakeNavigator());
        Assert.Equal(4, reopened.Panel.Options.Horizon);
        reopened.Panel.ResetToDefaultsCommand.Execute(null);
        reopened.History.RestoreCommand.Execute(reopened.History.Items[0]);
        Assert.True(reopened.HasResult);
        Assert.Equal(4, reopened.ForecastPoints.Count);
        Assert.Equal(4, reopened.Panel.Options.Horizon);
        Assert.Equal([0.25f, 0.5f, 0.75f], reopened.Panel.Options.QuantileLevels);
    }

    [Fact]
    public void Chart_RangeFitsNegativeAndFlatData()
    {
        var (min, max) = ForecastChart.ValueRange([-10, 10]);
        Assert.Equal(-12, min, 6);
        Assert.Equal(12, max, 6);
        var (flatMin, flatMax) = ForecastChart.ValueRange([5, 5]);
        Assert.True(flatMin < 5 && flatMax > 5);
    }
}
