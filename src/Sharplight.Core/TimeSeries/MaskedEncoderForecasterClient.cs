using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// The built-in <see cref="ITimeSeriesForecaster"/> over the masked-encoder form of <see cref="TimeSeriesGraphs"/> (Moirai 1.x), following uni2ts'
/// <c>MoiraiForecast</c> at a fixed patch size: one bidirectional pass over the context patches and the masked patches of the whole horizon (up to
/// <see cref="TimeSeriesGraphs.MaxHorizonKey"/> steps; <see cref="PredictionLength"/> is only the default horizon) gives a mixture distribution per step,
/// from which <see cref="TimeSeriesForecastOptions.SampleCount"/> values are drawn; the quantiles are the samples' empirical quantiles, the mean their mean.
/// Missing values are 0 with a false observation flag, as uni2ts' gluonts pipeline passes them; the patch size comes from
/// <see cref="TimeSeriesForecastOptions.PatchSize"/> (default <see cref="TimeSeriesGraphs.PatchSizeKey"/>), each size compiling its graph on first use.
/// <see cref="TimeSeriesForecastOptions.AutoPatchSize"/> follows uni2ts' <c>patch_size="auto"</c>: every patch size forecasts the history's last horizon
/// from the steps before it, and the one whose mixture gives those held-out values the lowest mean negative log density forecasts the request.
/// </summary>
internal sealed class MaskedEncoderForecasterClient : ITimeSeriesForecaster
{
    private static readonly float[] s_levels = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
    private readonly TaskModel _model;
    private readonly Dictionary<int, CompiledModel> _compiled = [];
    private readonly int[] _patchSizes;
    private readonly int _defaultPatchSize;
    private readonly int _maxHorizon;
    private readonly int _defaultSamples;
    private readonly MixtureDistribution _mixture;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public MaskedEncoderForecasterClient(TaskModel model)
    {
        _model = model;
        var component = model.Description.Component(ModelComponentRoles.Encoder);
        ContextLength = component.ConfigValue(TimeSeriesGraphs.ContextLengthKey, 0);
        PredictionLength = component.ConfigValue(TimeSeriesGraphs.PredictionLengthKey, 64);
        _defaultPatchSize = component.ConfigValue(TimeSeriesGraphs.PatchSizeKey, 32);
        _patchSizes = [.. component.ConfigValue(TimeSeriesGraphs.PatchSizesKey, "").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse)];
        _maxHorizon = component.ConfigValue(TimeSeriesGraphs.MaxHorizonKey, 2048);
        _defaultSamples = component.ConfigValue(TimeSeriesGraphs.SampleCountKey, 100);
        _mixture = new MixtureDistribution(component.ConfigValue(TimeSeriesGraphs.ComponentsKey, ""));
        Compiled(_defaultPatchSize);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int ContextLength { get; }

    public int PredictionLength { get; }

    public IReadOnlyList<float> QuantileLevels => s_levels;

    /// <summary>The patch sizes the model was trained with (<see cref="TimeSeriesForecastOptions.PatchSize"/>).</summary>
    public IReadOnlyList<int> PatchSizes => _patchSizes;

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<TimeSeriesForecast> ForecastAsync(ReadOnlyMemory<float> history, TimeSeriesForecastOptions? options = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        TimeSeriesHistory.Validate(history.Span);
        int horizon = CheckHorizon(options?.Horizon ?? PredictionLength);
        int samples = options?.SampleCount ?? _defaultSamples;
        ArgumentOutOfRangeException.ThrowIfLessThan(samples, 1, nameof(options));
        int patchSize = CheckPatchSize(options?.PatchSize);
        var levels = TimeSeriesQuantiles.Requested(options?.QuantileLevels, s_levels);
        var random = options?.Seed is { } seed ? new Random(seed) : new Random();
        var context = history.ToArray();
        return BackgroundCompute.RunAsync(_gate, () => Forecast(context, horizon, patchSize, samples, new VariateSampler(random), levels, cancellationToken), cancellationToken);
    }

    /// <summary>
    /// The predictive distribution of every step (for parity tests): the scaler's loc and scale and per step the mapped mixture parameters (layout of
    /// <see cref="MixtureDistribution.Map"/>) of the scaled series.
    /// </summary>
    internal Task<(float Loc, float Scale, double[][] Parameters)> DistributionAsync(float[] history, int horizon, int? patchSize = null)
    {
        TimeSeriesHistory.Validate(history);
        int size = CheckPatchSize(patchSize);
        CheckHorizon(horizon);
        return BackgroundCompute.RunAsync(_gate, () => Distribution(history, horizon,
            size == TimeSeriesForecastOptions.AutoPatchSize ? AutoPatchSize(history, horizon, CancellationToken.None) : size), CancellationToken.None);
    }

    /// <summary>
    /// The validation loss of every patch size (<see cref="PatchSizes"/> order) and the patch size <see cref="TimeSeriesForecastOptions.AutoPatchSize"/>
    /// picks for this history and horizon (for parity tests).
    /// </summary>
    internal Task<(double[] Losses, int PatchSize)> ValidationLossesAsync(float[] history, int horizon)
    {
        TimeSeriesHistory.Validate(history);
        CheckHorizon(horizon);
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var losses = _patchSizes.Select(ps => ValidationLoss(history, horizon, ps)).ToArray();
            return (losses, AutoPatchSize(history, horizon, CancellationToken.None));
        }, CancellationToken.None);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var compiled in _compiled.Values)
        {
            compiled.Dispose();
        }

        _gate.Dispose();
    }

    private int CheckHorizon(int horizon)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(horizon, 1, "options");
        if (horizon > _maxHorizon)
        {
            throw new ArgumentOutOfRangeException("options", $"The horizon {horizon} exceeds the model's longest horizon {_maxHorizon}.");
        }

        return horizon;
    }

    private int CheckPatchSize(int? patchSize)
    {
        int size = patchSize ?? _defaultPatchSize;
        if (size != TimeSeriesForecastOptions.AutoPatchSize && !_patchSizes.Contains(size))
        {
            throw new ArgumentOutOfRangeException("options", $"Patch size {size} is not one of the model's ({string.Join(", ", _patchSizes)}) or {TimeSeriesForecastOptions.AutoPatchSize} (auto).");
        }

        return size;
    }

    private CompiledModel Compiled(int patchSize)
    {
        if (!_compiled.TryGetValue(patchSize, out var compiled))
        {
            compiled = _model.CompileComponent(ModelComponentRoles.Encoder, TimeSeriesGraphs.MaskedEncoderGraph(patchSize));
            _compiled[patchSize] = compiled;
        }

        return compiled;
    }

    private TimeSeriesForecast Forecast(float[] history, int horizon, int patchSize, int sampleCount, VariateSampler sampler, IReadOnlyList<float> levels,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (patchSize == TimeSeriesForecastOptions.AutoPatchSize)
        {
            patchSize = AutoPatchSize(history, horizon, cancellationToken);
        }

        var (loc, scale, parameters) = Distribution(history, horizon, patchSize);
        var quantiles = new float[levels.Count][];
        for (int l = 0; l < levels.Count; l++)
        {
            quantiles[l] = new float[horizon];
        }

        var mean = new float[horizon];
        var column = new float[sampleCount];
        for (int t = 0; t < horizon; t++)
        {
            double sum = 0;
            for (int s = 0; s < sampleCount; s++)
            {
                // uni2ts' AffineTransformed: the scaled draw · scale + loc, in float.
                column[s] = ((float)_mixture.Sample(parameters[t], sampler) * scale) + loc;
                sum += column[s];
            }

            mean[t] = (float)(sum / sampleCount);
            Array.Sort(column);
            for (int l = 0; l < levels.Count; l++)
            {
                quantiles[l][t] = TimeSeriesQuantiles.Of(column, levels[l]);
            }
        }

        return new TimeSeriesForecast(levels, quantiles, mean);
    }

    private (float Loc, float Scale, double[][] Parameters) Distribution(float[] history, int horizon, int ps)
    {
        var window = history.Length <= ContextLength ? history : history[^ContextLength..];
        return Pass(window, horizon, 0, ps);
    }

    /// <summary>
    /// uni2ts' validation loss of patch size <paramref name="ps"/> (<c>MoiraiForecast._val_loss</c> with <c>SampleNLLLoss</c>): the history's last
    /// <see cref="ContextLength"/> + <paramref name="horizon"/> steps (left-padded when shorter), the first <see cref="ContextLength"/> as the context
    /// and the last <paramref name="horizon"/> as held-out targets; the mean negative log density of the observed targets under the predicted mixture
    /// (in the series' own scale), 0 when no target is observed.
    /// </summary>
    private double ValidationLoss(float[] history, int horizon, int ps)
    {
        int end = Math.Max(0, history.Length - horizon);
        var context = history[Math.Max(0, end - ContextLength)..end];
        int padding = Math.Max(0, horizon - history.Length);
        var (loc, scale, parameters) = Pass(context, horizon, padding / ps, ps);
        double logScale = Math.Log(scale);
        double sum = 0;
        int count = 0;
        for (int t = padding; t < horizon; t++)
        {
            float target = history[history.Length - horizon + t];
            if (float.IsNaN(target))
            {
                continue;
            }

            // AffineTransformed.log_prob: the base density at (y − loc) / scale (in float, as torch's inverse transform) less log scale.
            sum -= _mixture.LogProb(parameters[t - (padding / ps * ps)], (target - loc) / scale) - logScale;
            count++;
        }

        return count == 0 ? 0 : sum / count;
    }

    /// <summary>The patch size of the lowest validation loss (the first of equal ones, as torch's argmin), as uni2ts' <c>patch_size="auto"</c>.</summary>
    private int AutoPatchSize(float[] history, int horizon, CancellationToken cancellationToken)
    {
        int best = _patchSizes[0];
        double bestLoss = double.PositiveInfinity;
        foreach (int ps in _patchSizes)
        {
            cancellationToken.ThrowIfCancellationRequested();
            double loss = ValidationLoss(history, horizon, ps);
            if (loss < bestLoss)
            {
                best = ps;
                bestLoss = loss;
            }
        }

        return best;
    }

    /// <summary>
    /// One pass of the masked encoder: the context patches of <paramref name="context"/> (left-padded to whole patches) and the masked horizon patches
    /// <paramref name="skip"/> ... ⌈<paramref name="horizon"/> / <paramref name="ps"/>⌉ − 1 (earlier horizon patches are pure padding, which uni2ts
    /// keeps apart as another sample; their time ids stay reserved). Returns the scaler's loc and scale and the mapped mixture parameters of the horizon
    /// steps from patch <paramref name="skip"/> on.
    /// </summary>
    private (float Loc, float Scale, double[][] Parameters) Pass(float[] context, int horizon, int skip, int ps)
    {
        var compiled = Compiled(ps);
        var (values, observed) = MoiraiPatches.Pad(context, ps);
        var (loc, scale) = MoiraiPatches.Scale(values, observed);
        var times = MoiraiPatches.TimeIds(observed, ps);
        int contextRows = times.Length;
        int firstFuture = (contextRows == 0 ? 0 : times[^1]) + 1;
        int future = ((horizon + ps - 1) / ps) - skip;
        int rows = contextRows + future;
        var patches = new float[rows * ps];
        var predict = new float[rows];
        var positions = new int[rows];
        var spans = new int[rows * 2];
        var read = new int[future];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < ps; c++)
            {
                patches[(r * ps) + c] = MoiraiPatches.Scaled(r < contextRows ? values[(r * ps) + c] : 0f, loc, scale);
            }

            predict[r] = r < contextRows ? 0f : 1f;
            positions[r] = r < contextRows ? times[r] : firstFuture + skip + (r - contextRows);
            spans[(2 * r) + 1] = rows;
            if (r >= contextRows)
            {
                read[r - contextRows] = r;
            }
        }

        compiled.SetInput<float>(Input(compiled, TimeSeriesGraphs.PatchesInput), patches);
        compiled.SetInput<float>(Input(compiled, TimeSeriesGraphs.PredictMaskInput), predict);
        compiled.SetInput<int>(Input(compiled, TimeSeriesGraphs.PositionsInput), positions);
        compiled.SetInput<int>(Input(compiled, TimeSeriesGraphs.SpansInput), spans);
        compiled.SetInput<int>(Input(compiled, TimeSeriesGraphs.ReadRowsInput), read);
        var run = new RunParameters();
        run.Symbols[TimeSeriesGraphs.PatchSymbol] = rows;
        run.Symbols[TimeSeriesGraphs.ReadSymbol] = future;
        compiled.Run(run);
        int columns = _mixture.Columns;
        var output = new float[future * ps * columns];
        compiled.GetOutput<float>(TimeSeriesGraphs.DistributionOutput, output);
        var parameters = new double[horizon - (skip * ps)][];
        for (int t = 0; t < parameters.Length; t++)
        {
            parameters[t] = _mixture.Map(output.AsSpan(t * columns, columns));
        }

        return (loc, scale, parameters);
    }

    private static Graph.Value Input(CompiledModel compiled, string name) => compiled.Graph.Inputs.First(v => v.Input!.Name == name);
}
