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
        return BackgroundCompute.RunAsync(_gate, () => Distribution(history, horizon, size), CancellationToken.None);
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
        if (!_patchSizes.Contains(size))
        {
            throw new ArgumentOutOfRangeException("options", $"Patch size {size} is not one of the model's ({string.Join(", ", _patchSizes)}).");
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
        var compiled = Compiled(ps);
        var window = history.Length <= ContextLength ? history : history[^ContextLength..];
        var (values, observed) = MoiraiPatches.Pad(window, ps);
        var (loc, scale) = MoiraiPatches.Scale(values, observed);
        var times = MoiraiPatches.TimeIds(observed, ps);
        int context = times.Length;
        int future = (horizon + ps - 1) / ps;
        int rows = context + future;
        var patches = new float[rows * ps];
        var predict = new float[rows];
        var positions = new int[rows];
        var spans = new int[rows * 2];
        var read = new int[future];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < ps; c++)
            {
                patches[(r * ps) + c] = MoiraiPatches.Scaled(r < context ? values[(r * ps) + c] : 0f, loc, scale);
            }

            predict[r] = r < context ? 0f : 1f;
            positions[r] = r < context ? times[r] : times[^1] + 1 + (r - context);
            spans[(2 * r) + 1] = rows;
            if (r >= context)
            {
                read[r - context] = r;
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
        var parameters = new double[horizon][];
        for (int t = 0; t < horizon; t++)
        {
            parameters[t] = _mixture.Map(output.AsSpan(t * columns, columns));
        }

        return (loc, scale, parameters);
    }

    private static Graph.Value Input(CompiledModel compiled, string name) => compiled.Graph.Inputs.First(v => v.Input!.Name == name);
}
