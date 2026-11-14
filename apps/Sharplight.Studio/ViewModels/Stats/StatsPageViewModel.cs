using System.Collections.ObjectModel;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Controls;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;

namespace Sharplight.Studio.ViewModels.Stats;

/// <summary>The statistics dashboard: history from the store, plus live resource use while the page is visible.</summary>
internal sealed partial class StatsPageViewModel : PageViewModel
{
    private const string AllModels = "All models";
    private const int LiveSamples = 90;
    private const int RecentRows = 50;

    private readonly StudioServices _services;
    private readonly List<ChartPoint> _cpu = [];
    private readonly List<ChartPoint> _busy = [];
    private readonly List<ChartPoint> _throughput = [];
    private CancellationTokenSource? _live;
    private int _sampleIndex;

    public StatsPageViewModel(StudioServices services)
    {
        _services = services;
        SelectedRange = TimeRange.All[1];
        SelectedModel = AllModels;
        services.Stats.RecordAdded += (_, _) => services.Dispatcher.Post(() => _ = RefreshAsync());
    }

    /// <inheritdoc/>
    public override string Title => "Statistics";

    /// <inheritdoc/>
    public override string Icon => "IconStats";

    /// <summary>Range choices.</summary>
    public IReadOnlyList<TimeRange> Ranges => TimeRange.All;

    /// <summary>The selected range.</summary>
    [ObservableProperty]
    public partial TimeRange SelectedRange { get; set; }

    /// <summary>Model filter choices.</summary>
    public ObservableCollection<string> ModelChoices { get; } = [AllModels];

    /// <summary>The selected model filter.</summary>
    [ObservableProperty]
    public partial string SelectedModel { get; set; }

    /// <summary>Key figures.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<StatTile> Tiles { get; set; } = [];

    /// <summary>Decode speed per request over time.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> DecodeSeries { get; set; } = [];

    /// <summary>Prefill speed per request over time.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> PrefillSeries { get; set; } = [];

    /// <summary>Time to first token per request over time.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> TtftSeries { get; set; } = [];

    /// <summary>Tokens per time bucket (generated, prompt on top).</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartBar> TokenBars { get; set; } = [];

    /// <summary>Per-model aggregates.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ModelStatsRow> PerModel { get; set; } = [];

    /// <summary>Most recent requests.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<RequestRowViewModel> Recent { get; set; } = [];

    /// <summary>Process CPU over the last samples.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> CpuSeries { get; set; } = [];

    /// <summary>Engine busy share over the last samples.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> BusySeries { get; set; } = [];

    /// <summary>Generated tokens per second over the last samples.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> ThroughputSeries { get; set; } = [];

    /// <summary>Device memory bars.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<DeviceUsageViewModel> DeviceMemory { get; set; } = [];

    /// <summary>Current CPU use.</summary>
    [ObservableProperty]
    public partial string CpuText { get; set; } = "–";

    /// <summary>Current engine busy share.</summary>
    [ObservableProperty]
    public partial string BusyText { get; set; } = "–";

    /// <summary>Current throughput.</summary>
    [ObservableProperty]
    public partial string ThroughputText { get; set; } = "–";

    /// <summary>Process memory.</summary>
    [ObservableProperty]
    public partial string MemoryText { get; set; } = "–";

    /// <summary>Running and queued sequences.</summary>
    [ObservableProperty]
    public partial string SequencesText { get; set; } = "–";

    /// <summary>True when the range has no requests.</summary>
    [ObservableProperty]
    public partial bool IsEmpty { get; set; } = true;

    /// <inheritdoc/>
    public override async Task ActivateAsync()
    {
        await RefreshAsync();
        StartLive();
    }

    /// <inheritdoc/>
    public override void Deactivate()
    {
        _live?.Cancel();
        _live = null;
    }

    partial void OnSelectedRangeChanged(TimeRange value) => _ = RefreshAsync();

    partial void OnSelectedModelChanged(string value) => _ = RefreshAsync();

    /// <summary>Reloads the history.</summary>
    [RelayCommand]
    public async Task RefreshAsync()
    {
        var model = SelectedModel == AllModels ? null : SelectedModel;
        var since = DateTimeOffset.UtcNow - SelectedRange.Span;
        var all = await _services.Stats.QueryAsync(since, null);
        UpdateModelChoices(all);
        var records = model is null ? all : [.. all.Where(record => record.Model == model)];
        var summary = StatsSummary.From(records);
        IsEmpty = records.Count == 0;
        Tiles =
        [
            new("Requests", ByteFormat.Count(summary.Requests), "", "IconChat", $"Chat and API requests in the last {SelectedRange.Name}"),
            new("Generated", ByteFormat.Count(summary.CompletionTokens), "tokens", "IconSparkle", $"{ByteFormat.Count(summary.PromptTokens)} prompt tokens"),
            new("Decode", Format(summary.DecodeTokensPerSecond, "F1"), "tok/s", "IconBolt", "Generated tokens per second of decode time"),
            new("Prefill", Format(summary.PrefillTokensPerSecond, "F0"), "tok/s", "IconStats", "Computed prompt tokens per second of prefill time"),
            new("Time to first token", Format(summary.MedianTimeToFirstTokenMs, "F0"), "ms", "IconClock", $"Median; p95 {Format(summary.P95TimeToFirstTokenMs, "F0")} ms"),
            new("Draft acceptance", summary.AcceptanceRate is { } rate ? Format(rate * 100, "F0") : "–", summary.AcceptanceRate is null ? "" : "%",
                "IconCheck", "Speculative decoding: accepted / proposed draft tokens"),
        ];
        DecodeSeries = [.. records.Where(record => record.Stats.CompletionTokens > 1).Select(record => new ChartPoint(record.Timestamp.ToUnixTimeSeconds(), record.Stats.DecodeTokensPerSecond))];
        PrefillSeries = [.. records.Where(record => record.Stats.PrefillMs > 0).Select(record => new ChartPoint(record.Timestamp.ToUnixTimeSeconds(), record.Stats.PrefillTokensPerSecond))];
        TtftSeries = [.. records.Select(record => new ChartPoint(record.Timestamp.ToUnixTimeSeconds(), record.Stats.TimeToFirstTokenMs))];
        TokenBars = Bucket(records, since, SelectedRange);
        PerModel = [.. records.GroupBy(record => record.Model).OrderByDescending(group => group.Count()).Select(group =>
        {
            var s = StatsSummary.From([.. group]);
            return new ModelStatsRow(group.Key, s.Requests.ToString(CultureInfo.InvariantCulture), ByteFormat.Count(s.CompletionTokens),
                Format(s.DecodeTokensPerSecond, "F1") + " tok/s", Format(s.PrefillTokensPerSecond, "F0") + " tok/s", Format(s.MedianTimeToFirstTokenMs, "F0") + " ms");
        })];
        Recent = [.. records.TakeLast(RecentRows).Reverse().Select(record => new RequestRowViewModel(record))];
    }

    /// <summary>Takes one live sample (the page does this every second while visible).</summary>
    public void SampleNow()
    {
        var sample = _services.Monitor.Sample();
        int x = _sampleIndex++;
        Push(_cpu, new ChartPoint(x, sample.CpuPercent));
        Push(_busy, new ChartPoint(x, sample.EngineBusyPercent));
        Push(_throughput, new ChartPoint(x, sample.TokensPerSecond));
        CpuSeries = [.. _cpu];
        BusySeries = [.. _busy];
        ThroughputSeries = [.. _throughput];
        CpuText = Format(sample.CpuPercent, "F0") + " %";
        BusyText = Format(sample.EngineBusyPercent, "F0") + " %";
        ThroughputText = Format(sample.TokensPerSecond, "F1") + " tok/s";
        MemoryText = $"{ByteFormat.Format(sample.ProcessMemoryBytes)} of {ByteFormat.Format(sample.SystemMemoryBytes)}";
        SequencesText = $"{sample.ActiveSequences} running · {sample.QueueDepth} queued";
        DeviceMemory = [.. sample.Devices.Select(device => new DeviceUsageViewModel(device))];
    }

    [RelayCommand]
    private async Task ClearHistoryAsync()
    {
        await _services.Stats.ClearAsync();
        await RefreshAsync();
    }

    private void StartLive()
    {
        _live?.Cancel();
        var live = new CancellationTokenSource();
        _live = live;
        _ = RunLiveAsync(live.Token);
    }

    private async Task RunLiveAsync(CancellationToken cancellationToken)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(1));
        try
        {
            do
            {
                SampleNow();
            }
            while (await timer.WaitForNextTickAsync(cancellationToken));
        }
        catch (OperationCanceledException)
        {
            // The page was left.
        }
    }

    private void UpdateModelChoices(IReadOnlyList<RequestRecord> records)
    {
        foreach (var model in records.Select(record => record.Model).Distinct().Order(StringComparer.Ordinal))
        {
            if (!ModelChoices.Contains(model))
            {
                ModelChoices.Add(model);
            }
        }
    }

    /// <summary>Tokens per bucket from <paramref name="since"/> to now.</summary>
    internal static IReadOnlyList<ChartBar> Bucket(IReadOnlyList<RequestRecord> records, DateTimeOffset since, TimeRange range)
    {
        int count = (int)Math.Ceiling(range.Span / range.Bucket);
        var generated = new double[count];
        var prompt = new double[count];
        foreach (var record in records)
        {
            int index = (int)((record.Timestamp - since) / range.Bucket);
            if (index >= 0 && index < count)
            {
                generated[index] += record.Stats.CompletionTokens;
                prompt[index] += record.Stats.PromptTokens - record.Stats.CachedTokens;
            }
        }

        string format = range.Bucket >= TimeSpan.FromDays(1) ? "dd MMM" : "HH:mm";
        return [.. Enumerable.Range(0, count).Select(i =>
            new ChartBar((since + (range.Bucket * i)).ToLocalTime().ToString(format, CultureInfo.InvariantCulture), generated[i], prompt[i]))];
    }

    private static void Push(List<ChartPoint> series, ChartPoint point)
    {
        series.Add(point);
        if (series.Count > LiveSamples)
        {
            series.RemoveAt(0);
        }
    }

    private static string Format(double value, string format) => value.ToString(format, CultureInfo.InvariantCulture);
}
