using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// The built-in <see cref="ITimeSeriesForecaster"/> over the patch-quantile form of <see cref="TimeSeriesGraphs"/> (Chronos-Bolt). One pass forecasts
/// <see cref="PredictionLength"/> steps; longer horizons follow Chronos-Bolt's pipeline: every quantile forecast of the first pass extends its own copy of the
/// context, each copy is forecast again, and the empirical quantiles of all those forecasts give the next block.
/// </summary>
internal sealed class PatchQuantileForecasterClient : ITimeSeriesForecaster
{
    private const float ScaleEpsilon = 1e-5f;
    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _patchesInput;
    private readonly Graph.Value _maskInput;
    private readonly int _patchSize;
    private readonly int _patches;
    private readonly float[] _levels;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public PatchQuantileForecasterClient(TaskModel model)
    {
        _model = model;
        var component = model.Description.Component(ModelComponentRoles.Encoder);
        ContextLength = component.ConfigValue(TimeSeriesGraphs.ContextLengthKey, 0);
        PredictionLength = component.ConfigValue(TimeSeriesGraphs.PredictionLengthKey, 0);
        _patchSize = component.ConfigValue(TimeSeriesGraphs.PatchSizeKey, 0);
        _levels = TimeSeriesQuantiles.Parse(component.ConfigValue(TimeSeriesGraphs.QuantilesKey, ""));
        _patches = (ContextLength + _patchSize - 1) / _patchSize;
        _compiled = model.CompileComponent(ModelComponentRoles.Encoder, TimeSeriesGraphs.ForecastGraph);
        _patchesInput = _compiled.Graph.Inputs.First(v => v.Input!.Name == TimeSeriesGraphs.PatchesInput);
        _maskInput = _compiled.Graph.Inputs.First(v => v.Input!.Name == TimeSeriesGraphs.MaskInput);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int ContextLength { get; }

    public int PredictionLength { get; }

    public IReadOnlyList<float> QuantileLevels => _levels;

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<TimeSeriesForecast> ForecastAsync(ReadOnlyMemory<float> history, TimeSeriesForecastOptions? options = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        TimeSeriesHistory.Validate(history.Span);
        int horizon = options?.Horizon ?? PredictionLength;
        ArgumentOutOfRangeException.ThrowIfLessThan(horizon, 1, nameof(options));
        var levels = TimeSeriesQuantiles.Requested(options?.QuantileLevels, _levels);
        var context = history.ToArray();
        return BackgroundCompute.RunAsync(_gate, () => Forecast(context, horizon, levels, cancellationToken), cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _compiled.Dispose();
        _gate.Dispose();
    }

    private TimeSeriesForecast Forecast(float[] history, int horizon, IReadOnlyList<float> levels, CancellationToken cancellationToken)
    {
        int count = _levels.Length;
        var context = Tail(history, ContextLength);
        var prediction = RunPass(context);
        var blocks = new List<float[][]> { prediction };
        int remaining = horizon - PredictionLength;
        var contexts = remaining > 0 ? Enumerable.Repeat(context, count).ToArray() : [];
        var samples = new float[count * count];
        while (remaining > 0)
        {
            var passes = new float[count][][];
            for (int q = 0; q < count; q++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                contexts[q] = Tail([.. contexts[q], .. prediction[q]], ContextLength);
                passes[q] = RunPass(contexts[q]);
            }

            // The quantiles of all count² forecasts of a step become the next block.
            var next = new float[count][];
            for (int q = 0; q < count; q++)
            {
                next[q] = new float[PredictionLength];
            }

            for (int t = 0; t < PredictionLength; t++)
            {
                for (int a = 0; a < count; a++)
                {
                    for (int b = 0; b < count; b++)
                    {
                        samples[(a * count) + b] = passes[a][b][t];
                    }
                }

                Array.Sort(samples);
                for (int q = 0; q < count; q++)
                {
                    next[q][t] = TimeSeriesQuantiles.Of(samples, _levels[q]);
                }
            }

            prediction = next;
            blocks.Add(prediction);
            remaining -= PredictionLength;
        }

        var native = new float[count][];
        for (int q = 0; q < count; q++)
        {
            native[q] = new float[horizon];
            for (int t = 0; t < horizon; t++)
            {
                native[q][t] = blocks[t / PredictionLength][q][t % PredictionLength];
            }
        }

        int median = TimeSeriesQuantiles.IndexOf(_levels, 0.5f);
        var mean = median >= 0 ? (float[])native[median].Clone() : TimeSeriesQuantiles.Select(_levels, native, [0.5f])[0];
        return new TimeSeriesForecast(levels, TimeSeriesQuantiles.Select(_levels, native, levels), mean);
    }

    private static float[] Tail(float[] values, int length) => values.Length <= length ? values : values[^length..];

    // One model pass over at most ContextLength observations: the unscaled forecast [levels][PredictionLength].
    private float[][] RunPass(float[] context)
    {
        var (loc, scale) = TimeSeriesHistory.InstanceScale(context, ScaleEpsilon);
        int width = 2 * _patchSize;
        var patches = new float[_patches * width];
        var mask = new float[_patches];
        int offset = (_patches * _patchSize) - context.Length;   // left padding with missing values
        for (int i = 0; i < context.Length; i++)
        {
            float value = context[i];
            if (float.IsNaN(value))
            {
                continue;
            }

            int position = offset + i;
            int patch = position / _patchSize;
            int column = position % _patchSize;
            patches[(patch * width) + column] = (float)((value - loc) / scale);
            patches[(patch * width) + _patchSize + column] = 1f;
            mask[patch] = 1f;
        }

        _compiled.SetInput<float>(_patchesInput, patches);
        _compiled.SetInput<float>(_maskInput, mask);
        _compiled.Run(new RunParameters());
        var output = new float[_levels.Length * PredictionLength];
        _compiled.GetOutput<float>(TimeSeriesGraphs.QuantilesOutput, output);
        var result = new float[_levels.Length][];
        for (int q = 0; q < _levels.Length; q++)
        {
            result[q] = new float[PredictionLength];
            for (int t = 0; t < PredictionLength; t++)
            {
                result[q][t] = (float)((output[(q * PredictionLength) + t] * scale) + loc);
            }
        }

        return result;
    }
}
