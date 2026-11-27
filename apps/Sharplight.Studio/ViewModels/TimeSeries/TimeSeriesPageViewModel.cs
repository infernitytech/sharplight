using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Models;
using Sharplight.Studio.Controls;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.TimeSeries;

namespace Sharplight.Studio.ViewModels.TimeSeries;

/// <summary>
/// The Time series page (kind <c>time-series</c>): a CSV or TSV file is read into its numeric columns, the chosen column goes to the model's
/// <see cref="ITimeSeriesForecaster"/>, and the forecast is drawn after the history with a shaded prediction interval, listed per step and exported as
/// CSV. Built from the shared inference blocks (picker, parameters, queue, history, outputs).
/// </summary>
internal sealed partial class TimeSeriesPageViewModel : InferencePageViewModel
{
    private const string ForecastName = "forecast.json";
    private const string CsvName = "forecast.csv";
    private const int DefaultHorizon = 24;
    private const int MaxTableRows = 200;
    private static readonly FileFilter s_tableFiles = new("Tables", ["*.csv", "*.tsv", "*.txt"]);
    private static readonly FileFilter s_allFiles = new("All files", ["*"]);

    private TimeSeriesTable? _table;
    private ForecastFile? _result;
    private int _predictionLength;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    public TimeSeriesPageViewModel(StudioServices services, INavigator navigator)
        : base(services, navigator, ModelKinds.TimeSeriesForecasting, "time-series", CreateParameters(services))
    {
    }

    /// <inheritdoc/>
    public override string Title => "Time series";

    /// <inheritdoc/>
    public override string Icon => "IconStats";

    /// <inheritdoc/>
    public override string KindNoun => "time series forecasting";

    /// <summary>The typed parameter panel (maps to <see cref="TimeSeriesForecastOptions"/>).</summary>
    public ParameterPanelViewModel<TimeSeriesForecastOptions> Panel => (ParameterPanelViewModel<TimeSeriesForecastOptions>)Parameters;

    /// <summary>The file shown as the input.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(TableName))]
    public partial string TablePath { get; set; } = "";

    /// <summary>The name of the input file.</summary>
    public string TableName => Path.GetFileName(TablePath);

    /// <summary>Rows, columns and time column of the input.</summary>
    [ObservableProperty]
    public partial string TableSummary { get; set; } = "";

    /// <summary>The numeric columns of the input.</summary>
    public ObservableCollection<string> ColumnNames { get; } = [];

    /// <summary>The column that is forecast (the first numeric column once a file is read).</summary>
    [ObservableProperty]
    public partial string? SelectedColumn { get; set; }

    /// <summary>The other numeric columns, each of which can be chosen as a covariate (listed when <see cref="SupportsCovariates"/>).</summary>
    public ObservableCollection<CovariateColumnViewModel> Covariates { get; } = [];

    /// <summary>The chosen model uses covariates (Chronos-2); known once the model has been probed.</summary>
    [ObservableProperty]
    public partial bool SupportsCovariates { get; set; }

    /// <summary>The page is forecasting (a run is queued or running).</summary>
    public bool IsForecasting => Queue.IsBusy;

    /// <summary>The input has been read.</summary>
    public bool HasTable => _table is not null;

    /// <summary>The model's context and prediction length, once known.</summary>
    [ObservableProperty]
    public partial string ModelInfo { get; set; } = "";

    /// <summary>Why <see cref="RunCommand"/> cannot run, for the hint under the button; empty when it can.</summary>
    public string RunHint => !HasModel ? "Load a time series model first." : !HasTable ? "Choose a CSV file first." : "";

    /// <summary>The history that was drawn (the last observations).</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> HistoryPoints { get; set; } = [];

    /// <summary>The point forecast, after the history.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> ForecastPoints { get; set; } = [];

    /// <summary>The lower edge of the interval.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> LowerPoints { get; set; } = [];

    /// <summary>The upper edge of the interval.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ChartPoint> UpperPoints { get; set; } = [];

    /// <summary>The label under the left end of the chart.</summary>
    [ObservableProperty]
    public partial string ChartStart { get; set; } = "";

    /// <summary>The label under the right end of the chart.</summary>
    [ObservableProperty]
    public partial string ChartEnd { get; set; } = "";

    /// <summary>The steps of the forecast (the first <see cref="MaxTableRows"/>).</summary>
    public ObservableCollection<ForecastRowViewModel> Rows { get; } = [];

    /// <summary>Heading of the lower interval column, for example <c>10 %</c>.</summary>
    [ObservableProperty]
    public partial string LowHeader { get; set; } = "";

    /// <summary>Heading of the upper interval column, for example <c>90 %</c>.</summary>
    [ObservableProperty]
    public partial string HighHeader { get; set; } = "";

    /// <summary>The steps are timestamps (the table's time column continues with a constant step).</summary>
    [ObservableProperty]
    public partial bool HasTimes { get; set; }

    /// <summary>Horizon, interval and model of the shown forecast.</summary>
    [ObservableProperty]
    public partial string ResultSummary { get; set; } = "";

    /// <summary>There is a forecast to show.</summary>
    public bool HasResult => _result is not null;

    /// <summary>The task that reads the chosen model's context and prediction length (completed when there is none to read).</summary>
    internal Task ModelProbe { get; private set; } = Task.CompletedTask;

    /// <summary>
    /// Takes <paramref name="path"/> as the input: reads the table (CSV, TSV or semicolon separated), lists its numeric columns and chooses the first.
    /// A file that cannot be read, or has no numeric column, is refused with a notice.
    /// </summary>
    /// <param name="path">The file.</param>
    /// <returns>Whether the file is usable.</returns>
    public async Task<bool> SetTableAsync(string path)
    {
        Notice = "";
        if (!File.Exists(path))
        {
            Notice = $"{path} does not exist.";
            return false;
        }

        try
        {
            var table = await Task.Run(() => TimeSeriesTable.Load(path));
            _table = table;
            TablePath = path;
            ColumnNames.Clear();
            foreach (var column in table.Columns)
            {
                ColumnNames.Add(column.Name);
            }

            SelectedColumn = ColumnNames.FirstOrDefault();
            RebuildCovariates(clear: true);
            string columns = table.Columns.Count == 1 ? "1 numeric column" : $"{table.Columns.Count} numeric columns";
            TableSummary = $"{table.RowCount} rows · {columns}" + (table.Timestamps is null ? "" : table.TimeColumn is { Length: > 0 } time ? $" · time column {time}" : " · time column");
            OnInputChanged();
            return true;
        }
        catch (Exception error) when (error is IOException or FormatException or UnauthorizedAccessException or InvalidDataException)
        {
            Notice = $"{Path.GetFileName(path)} cannot be read: {error.Message}";
            return false;
        }
    }

    /// <inheritdoc/>
    protected override void OnModelChanged(ILoadedModel? model)
    {
        ModelInfo = "";
        SupportsCovariates = false;
        ModelProbe = model is { State: ResidentState.Ready } ? ProbeAsync(model) : Task.CompletedTask;
        OnInputChanged();
    }

    /// <inheritdoc/>
    protected override void OnBusyChanged()
    {
        OnPropertyChanged(nameof(IsForecasting));
        RunCommand.NotifyCanExecuteChanged();
    }

    /// <inheritdoc/>
    protected override void Restore(RunRecord record)
    {
        string path = Archive.PathOf(record, ForecastName);
        if (!File.Exists(path))
        {
            Notice = "The stored forecast is gone.";
            return;
        }

        var file = PageJson.Read(path, StudioJsonContext.Default.ForecastFile);
        Panel.Apply(record.Options);
        Show(file);
        if (record.Input.TryGetValue("file", out var name))
        {
            TablePath = name;
        }
    }

    partial void OnSelectedColumnChanged(string? value)
    {
        RebuildCovariates(clear: false);
        OnInputChanged();
    }

    /// <summary>Lists every numeric column except the target as a covariate choice, keeping the choices already made.</summary>
    private void RebuildCovariates(bool clear)
    {
        var previous = clear ? [] : Covariates.ToDictionary(covariate => covariate.Name);
        Covariates.Clear();
        foreach (string name in CovariateNames(ColumnNames, SelectedColumn))
        {
            var covariate = new CovariateColumnViewModel(name);
            if (previous.TryGetValue(name, out var old))
            {
                covariate.IsSelected = old.IsSelected;
                covariate.KnownInFuture = old.KnownInFuture;
            }

            Covariates.Add(covariate);
        }
    }

    /// <summary>The columns that can be covariates of <paramref name="target"/>: all but the target itself.</summary>
    internal static IReadOnlyList<string> CovariateNames(IEnumerable<string> columns, string? target)
        => [.. columns.Where(name => !string.Equals(name, target, StringComparison.Ordinal))];

    private void OnInputChanged()
    {
        OnPropertyChanged(nameof(HasTable));
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
    }

    private bool CanRun() => HasModel && HasTable && SelectedColumn is not null;

    /// <summary>Reads the context and prediction length of the model and, when the horizon is untouched, makes the prediction length the horizon.</summary>
    private async Task ProbeAsync(ILoadedModel model)
    {
        try
        {
            using var lease = await model.AcquireClientAsync<ITimeSeriesForecaster>(CancellationToken.None).ConfigureAwait(false);
            int context = lease.Client.ContextLength;
            int prediction = lease.Client.PredictionLength;
            bool covariates = lease.Client.SupportsCovariates;
            Services.Dispatcher.Post(() =>
            {
                if (!ReferenceEquals(model, SelectedModel))
                {
                    return;
                }

                ModelInfo = ModelText(context, prediction);
                SupportsCovariates = covariates;
                _predictionLength = prediction;
                if (!Panel.IsModified && Panel.GetInt("horizon") == DefaultHorizon)
                {
                    Panel.Apply(new Dictionary<string, string> { ["horizon"] = Math.Clamp(prediction, 1, 1024).ToString(CultureInfo.InvariantCulture) });
                }
            });
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            Services.Dispatcher.Post(() => ModelInfo = "");
        }
    }

    private static string ModelText(int context, int prediction)
        => $"context {context} · prediction length {prediction}";

    /// <summary>Opens the file picker for the input table.</summary>
    [RelayCommand]
    private async Task ChooseFileAsync()
    {
        var files = await Services.Platform.PickFilesAsync("Choose a CSV file", [s_tableFiles, s_allFiles], allowMultiple: false);
        if (files.Count > 0)
        {
            await SetTableAsync(files[0]);
        }
    }

    /// <summary>Forecasts the chosen column with the chosen model and the current parameters.</summary>
    [RelayCommand(CanExecute = nameof(CanRun))]
    private void Run()
    {
        var model = SelectedModel!;
        var table = _table!;
        var column = table.Column(SelectedColumn!);
        var options = Panel.Options;
        var values = Panel.Snapshot();
        string name = Path.GetFileName(TablePath);
        string title = $"{name} · {column.Name}";
        var chosen = SupportsCovariates
            ? Covariates.Where(covariate => covariate.IsSelected).Select(covariate => (table.Column(covariate.Name), covariate.KnownInFuture)).ToList()
            : [];
        int? horizon = options.Horizon is { } set && set != DefaultHorizon && set != _predictionLength ? set : null;
        Panel.Persist();
        Notice = "";
        Queue.Enqueue(title, model.Id, context => ForecastAsync(model, table, column, options, values, name, title, chosen, horizon, context));
        OnPropertyChanged(nameof(IsForecasting));
        RunCommand.NotifyCanExecuteChanged();
    }

    private async Task ForecastAsync(ILoadedModel model, TimeSeriesTable table, TimeSeriesColumn column, TimeSeriesForecastOptions options, Dictionary<string, string> values,
        string fileName, string title, IReadOnlyList<(TimeSeriesColumn Column, bool KnownFuture)> covariates, int? horizon, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        ForecastFile? file = null;
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        try
        {
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireClientAsync<ITimeSeriesForecaster>(context.CancellationToken).ConfigureAwait(false);
            context.Report(null, "Forecasting");
            var forecaster = lease.Client;
            if (covariates.Count > 0 && forecaster.SupportsCovariates)
            {
                var (request, resolved, length) = BuildRequest(column, covariates, horizon, options.Horizon ?? forecaster.PredictionLength);
                var multivariate = await forecaster.ForecastAsync(request, options with { Horizon = resolved }, context.CancellationToken).ConfigureAwait(false);
                var history = new TimeSeriesColumn(column.Name, column.Values[..length]);
                file = BuildFile(table, history, multivariate.Targets[0], options, forecaster);
            }
            else
            {
                var forecast = await forecaster.ForecastAsync(column.Values, options, context.CancellationToken).ConfigureAwait(false);
                file = BuildFile(table, column, forecast, options, forecaster);
            }
        }
        catch (OperationCanceledException)
        {
            outcome = RunOutcome.Cancelled;
        }
        catch (Exception failure) when (failure is not OutOfMemoryException)
        {
            outcome = RunOutcome.Failed;
            error = failure.GetBaseException().Message;
        }

        double seconds = watch.Elapsed.TotalSeconds;
        var metadata = new Dictionary<string, string> { ["column"] = column.Name, ["time"] = $"{seconds:0.0} s" };
        if (covariates.Count > 0 && file is not null)
        {
            metadata["covariates"] = string.Join(", ", covariates.Select(covariate => covariate.Column.Name));
        }

        if (file is not null)
        {
            metadata["horizon"] = file.Horizon.ToString(CultureInfo.InvariantCulture);
            metadata["interval"] = IntervalText(file);
            Services.Dispatcher.Post(() => Show(file));
        }

        var record = new RunRecord
        {
            Id = PageArchive.NewId(),
            Title = title,
            Model = model.Id,
            Outcome = outcome,
            Seconds = seconds,
            Error = error,
            Input = new Dictionary<string, string> { ["file"] = fileName, ["column"] = column.Name },
            Options = values,
            Metadata = metadata,
            Preview = file is null ? null : Summary(file),
            Files = file is null ? [] : [CsvName, ForecastName],
        };
        var files = file is null ? null : new Dictionary<string, byte[]>
        {
            [CsvName] = new System.Text.UTF8Encoding(false).GetBytes(file.ToCsv()),
            [ForecastName] = JsonSerializer.SerializeToUtf8Bytes(file, StudioJsonContext.Default.ForecastFile),
        };
        if (outcome != RunOutcome.Cancelled)
        {
            Archived(record, files);
        }

        if (outcome == RunOutcome.Cancelled)
        {
            throw new OperationCanceledException(context.CancellationToken);
        }

        if (error is not null)
        {
            Services.Dispatcher.Post(() => Notice = error);
            throw new InvalidOperationException(error);
        }
    }

    /// <summary>
    /// Builds the multivariate request of a table: the target's history is its rows up to the last observed value; a covariate marked known in the future takes the
    /// rows after that (the target's trailing empty rows) as its future when it has values there, a past-only covariate (or one without values there) has none.
    /// The horizon is <paramref name="horizon"/> when set, else the number of trailing rows of the known-future covariates, else <paramref name="defaultHorizon"/>;
    /// futures are padded with <see cref="float.NaN"/> or truncated to it.
    /// </summary>
    /// <param name="target">The column to forecast.</param>
    /// <param name="covariates">The chosen covariate columns and whether each is known in the future.</param>
    /// <param name="horizon">The horizon the user set, or <see langword="null"/>.</param>
    /// <param name="defaultHorizon">The horizon when neither the user nor the covariates give one.</param>
    /// <returns>The request, its horizon and the length of the history (the rows before the trailing empty ones).</returns>
    internal static (TimeSeriesForecastRequest Request, int Horizon, int HistoryLength) BuildRequest(TimeSeriesColumn target,
        IReadOnlyList<(TimeSeriesColumn Column, bool KnownFuture)> covariates, int? horizon, int defaultHorizon)
    {
        int length = target.Values.Length;
        while (length > 0 && !float.IsFinite(target.Values[length - 1]))
        {
            length--;
        }

        int trailing = target.Values.Length - length;
        var futures = covariates.Select(covariate => covariate.KnownFuture && trailing > 0 && covariate.Column.Values.Skip(length).Any(float.IsFinite)
            ? covariate.Column.Values[length..]
            : []).ToList();
        int resolved = horizon ?? (futures.Any(future => future.Length > 0) ? trailing : defaultHorizon);
        var list = new List<TimeSeriesCovariate>();
        for (int i = 0; i < covariates.Count; i++)
        {
            float[] future = futures[i];
            if (future.Length > 0)
            {
                var fitted = new float[resolved];
                Array.Fill(fitted, float.NaN);
                Array.Copy(future, fitted, Math.Min(future.Length, resolved));
                future = fitted;
            }

            list.Add(new TimeSeriesCovariate { Name = covariates[i].Column.Name, History = covariates[i].Column.Values.AsMemory(0, length), Future = future });
        }

        var request = new TimeSeriesForecastRequest { Targets = [target.Values.AsMemory(0, length)], Covariates = list };
        return (request, resolved, length);
    }

    /// <summary>Builds the stored forecast: the plotted tail of the history (about 4 horizons), the quantiles and the continued timestamps.</summary>
    internal static ForecastFile BuildFile(TimeSeriesTable table, TimeSeriesColumn column, TimeSeriesForecast forecast, TimeSeriesForecastOptions options, ITimeSeriesForecaster model)
    {
        var levels = options.QuantileLevels ?? forecast.QuantileLevels;
        float low = levels.Count > 0 ? levels[0] : forecast.QuantileLevels[0];
        float high = levels.Count > 0 ? levels[^1] : forecast.QuantileLevels[^1];
        int tail = Math.Min(column.Values.Length, Math.Max(16, 4 * forecast.Horizon));
        int first = column.Values.Length - tail;
        var stamps = table.Timestamps is { } all && all.Count >= column.Values.Length ? all.Take(column.Values.Length).ToList() : null;
        return new ForecastFile
        {
            Column = column.Name,
            TimeHeader = table.TimeColumn ?? "",
            History = [.. column.Values.Skip(first).Select(value => float.IsFinite(value) ? value : (float?)null)],
            HistoryTimes = stamps is not null ? [.. stamps.Skip(first)] : null,
            Levels = [.. forecast.QuantileLevels],
            Quantiles = [.. forecast.Quantiles],
            Mean = forecast.Mean,
            Times = ForecastTimeline.Continue(stamps ?? table.Timestamps, forecast.Horizon),
            LowLevel = low,
            HighLevel = high,
            ContextLength = model.ContextLength,
            PredictionLength = model.PredictionLength,
        };
    }

    private static string IntervalText(ForecastFile file)
        => $"{(file.HighLevel - file.LowLevel) * 100:0.#} %";

    private static string Summary(ForecastFile file)
        => $"{file.Horizon} steps of {file.Column}, {IntervalText(file)} interval";

    private void Show(ForecastFile file)
    {
        _result = file;
        int count = file.History.Count;
        HistoryPoints = [.. file.History.Select((value, index) => new ChartPoint(index, value ?? double.NaN))];
        var low = Nearest(file, file.LowLevel);
        var high = Nearest(file, file.HighLevel);
        IReadOnlyList<ChartPoint> At(IReadOnlyList<float> series) => [.. series.Select((value, step) => new ChartPoint(count + step, value))];
        ForecastPoints = At(file.Mean);
        LowerPoints = At(low);
        UpperPoints = At(high);
        ChartStart = file.HistoryTimes is { Count: > 0 } before ? before[0] : $"-{count}";
        ChartEnd = file.Times is { Count: > 0 } after ? after[^1] : $"+{file.Horizon}";
        LowHeader = $"{file.LowLevel * 100:0.#} %";
        HighHeader = $"{file.HighLevel * 100:0.#} %";
        HasTimes = file.Times is { Count: > 0 };
        Rows.Clear();
        for (int step = 0; step < Math.Min(file.Horizon, MaxTableRows); step++)
        {
            Rows.Add(new ForecastRowViewModel(step + 1, file.Times?[step] ?? "", low[step], file.Mean[step], high[step]));
        }

        ResultSummary = $"{file.Column} · {file.Horizon} steps · {IntervalText(file)} interval ({LowHeader} to {HighHeader})"
            + (file.ContextLength > 0 ? $" · model {ModelText(file.ContextLength, file.PredictionLength)}" : "");
        OnPropertyChanged(nameof(HasResult));
    }

    private static float[] Nearest(ForecastFile file, float level)
    {
        int best = 0;
        for (int i = 1; i < file.Levels.Count; i++)
        {
            if (Math.Abs(file.Levels[i] - level) < Math.Abs(file.Levels[best] - level))
            {
                best = i;
            }
        }

        return file.Quantiles[best];
    }

    /// <summary>Saves the forecast as CSV where the user chooses.</summary>
    [RelayCommand]
    private async Task ExportCsvAsync()
    {
        if (_result is not { } file)
        {
            return;
        }

        string stem = Path.GetFileNameWithoutExtension(TablePath);
        await Exporter.SaveTextAsync(Services.Platform, "Export forecast", $"{(stem.Length == 0 ? "forecast" : stem)}-forecast.csv", file.ToCsv());
    }

    /// <summary>Cancels the queued and running forecasts.</summary>
    [RelayCommand]
    private void Cancel() => Queue.CancelAll();

    private static ParameterPanelViewModel<TimeSeriesForecastOptions> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new NumberParameterViewModel("horizon", "Horizon", "Steps to forecast. Beyond the model's prediction length it predicts step by step, and the quality drops.", 1, 1024, 1, DefaultHorizon),
            new ChoiceParameterViewModel("interval", "Interval", "The shaded band holds this share of the likely values.",
                [.. ForecastInterval.Coverages.Select(percent => new ParameterChoice(percent.ToString(CultureInfo.InvariantCulture), $"{percent} %"))],
                ForecastInterval.DefaultCoverage.ToString(CultureInfo.InvariantCulture)),
            new NumberParameterViewModel("samples", "Samples", "Sample paths of sampling models (Chronos, Moirai 1.x); more are steadier and slower. Quantile models ignore it.", 1, 200, 1, 20),
            new TextParameterViewModel("seed", "Seed", "Fixes the samples of sampling models; empty is random.", "Random"),
        ];
        ParameterPreset[] presets =
        [
            new("Narrow interval", new() { ["interval"] = "50" }, BuiltIn: true),
            new("Default", new() { ["interval"] = "80", ["samples"] = "20" }, BuiltIn: true),
            new("Wide interval", new() { ["interval"] = "90", ["samples"] = "50" }, BuiltIn: true),
        ];
        return new ParameterPanelViewModel<TimeSeriesForecastOptions>(services.Pages, "time-series", parameters, panel =>
        {
            var (low, high) = ForecastInterval.Levels(int.Parse(panel.GetChoice("interval"), CultureInfo.InvariantCulture));
            _ = SeedParser.TryRead(panel.GetText("seed"), out long? seed);
            return new TimeSeriesForecastOptions
            {
                Horizon = panel.GetInt("horizon"),
                QuantileLevels = [low, 0.5f, high],
                SampleCount = panel.GetInt("samples"),
                Seed = seed is { } value ? (int)(value % int.MaxValue) : null,
            };
        }, presets);
    }
}
