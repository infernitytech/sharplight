using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// The built-in <see cref="ITimeSeriesForecaster"/> over the patch-quantile form of <see cref="TimeSeriesGraphs"/> (Chronos-Bolt). One pass forecasts
/// <see cref="PredictionLength"/> steps; longer horizons follow Chronos-Bolt's pipeline: every quantile forecast of the first pass extends its own copy of the
/// context, each copy is forecast again, and the empirical quantiles of all those forecasts give the next block. Graphs with a series axis
/// (<see cref="TimeSeriesGraphs.SeriesSymbol"/>) run up to <see cref="MaxSeriesPerPass"/> contexts per pass: the series of a batch request and the context copies of
/// a long horizon; older stored graphs run one context per pass.
/// </summary>
internal sealed class PatchQuantileForecasterClient : ITimeSeriesForecaster
{
    /// <summary>The most contexts one pass of a batched graph runs (larger batches take several passes).</summary>
    public const int MaxSeriesPerPass = 64;

    private const float ScaleEpsilon = 1e-5f;
    private readonly TaskModel _model;
    private readonly bool _batched;
    private readonly int _patchSize;
    private readonly int _patches;
    private readonly float[] _levels;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private CompiledModel _compiled = null!;
    private Graph.Value _patchesInput = null!;
    private Graph.Value _maskInput = null!;
    private int _seriesBound;
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
        _batched = model.Definition.Graph(TimeSeriesGraphs.ForecastGraph).Inputs
            .First(v => v.Input!.Name == TimeSeriesGraphs.PatchesInput).Rank == 3;

        // The context copies of a long horizon (one per quantile level) fit one pass from the start.
        Compile(_batched ? Math.Min(_levels.Length, MaxSeriesPerPass) : 1);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int ContextLength { get; }

    public int PredictionLength { get; }

    public IReadOnlyList<float> QuantileLevels => _levels;

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public async Task<TimeSeriesForecast> ForecastAsync(ReadOnlyMemory<float> history, TimeSeriesForecastOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        TimeSeriesHistory.Validate(history.Span);
        var forecasts = await ForecastBatchAsync([history], options, cancellationToken).ConfigureAwait(false);
        return forecasts[0];
    }

    public Task<IReadOnlyList<TimeSeriesForecast>> ForecastBatchAsync(IReadOnlyList<ReadOnlyMemory<float>> histories, TimeSeriesForecastOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        TimeSeriesHistory.Validate(histories);
        int horizon = options?.Horizon ?? PredictionLength;
        ArgumentOutOfRangeException.ThrowIfLessThan(horizon, 1, nameof(options));
        var levels = TimeSeriesQuantiles.Requested(options?.QuantileLevels, _levels);
        var contexts = histories.Select(history => history.ToArray()).ToArray();
        return BackgroundCompute.RunAsync(_gate, () => Forecast(contexts, horizon, levels, cancellationToken), cancellationToken);
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

    private IReadOnlyList<TimeSeriesForecast> Forecast(float[][] histories, int horizon, IReadOnlyList<float> levels, CancellationToken cancellationToken)
    {
        int count = _levels.Length;
        int seriesCount = histories.Length;
        var contexts = histories.Select(history => Tail(history, ContextLength)).ToArray();
        var predictions = RunPasses(contexts, cancellationToken);
        var blocks = predictions.Select(prediction => new List<float[][]> { prediction }).ToArray();
        int remaining = horizon - PredictionLength;
        var copies = remaining > 0 ? new float[seriesCount * count][] : [];
        for (int s = 0; s < seriesCount && remaining > 0; s++)
        {
            for (int q = 0; q < count; q++)
            {
                copies[(s * count) + q] = contexts[s];
            }
        }

        var samples = new float[count * count];
        while (remaining > 0)
        {
            for (int s = 0; s < seriesCount; s++)
            {
                for (int q = 0; q < count; q++)
                {
                    int copy = (s * count) + q;
                    copies[copy] = Tail([.. copies[copy], .. predictions[s][q]], ContextLength);
                }
            }

            var passes = RunPasses(copies, cancellationToken);
            for (int s = 0; s < seriesCount; s++)
            {
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
                            samples[(a * count) + b] = passes[(s * count) + a][b][t];
                        }
                    }

                    Array.Sort(samples);
                    for (int q = 0; q < count; q++)
                    {
                        next[q][t] = TimeSeriesQuantiles.Of(samples, _levels[q]);
                    }
                }

                predictions[s] = next;
                blocks[s].Add(next);
            }

            remaining -= PredictionLength;
        }

        var forecasts = new TimeSeriesForecast[seriesCount];
        int median = TimeSeriesQuantiles.IndexOf(_levels, 0.5f);
        for (int s = 0; s < seriesCount; s++)
        {
            var native = new float[count][];
            for (int q = 0; q < count; q++)
            {
                native[q] = new float[horizon];
                for (int t = 0; t < horizon; t++)
                {
                    native[q][t] = blocks[s][t / PredictionLength][q][t % PredictionLength];
                }
            }

            var mean = median >= 0 ? (float[])native[median].Clone() : TimeSeriesQuantiles.Select(_levels, native, [0.5f])[0];
            forecasts[s] = new TimeSeriesForecast(levels, TimeSeriesQuantiles.Select(_levels, native, levels), mean);
        }

        return forecasts;
    }

    private static float[] Tail(float[] values, int length) => values.Length <= length ? values : values[^length..];

    // The unscaled forecast [levels][PredictionLength] of every context (each at most ContextLength observations), as many per pass as the graph takes.
    private float[][][] RunPasses(IReadOnlyList<float[]> contexts, CancellationToken cancellationToken)
    {
        var results = new float[contexts.Count][][];
        int perPass = _batched ? MaxSeriesPerPass : 1;
        for (int first = 0; first < contexts.Count; first += perPass)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int n = Math.Min(perPass, contexts.Count - first);
            if (n > _seriesBound)
            {
                // Grow in powers of two, so a server's varying batch sizes recompile only a few times.
                _compiled.Dispose();
                Compile(Math.Min((int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)n), MaxSeriesPerPass));
            }

            RunPass(contexts, first, n, results);
        }

        return results;
    }

    // One model pass over contexts[first .. first + n].
    private void RunPass(IReadOnlyList<float[]> contexts, int first, int n, float[][][] results)
    {
        int width = 2 * _patchSize;
        var patches = new float[n * _patches * width];
        var mask = new float[n * _patches];
        var scales = new (double Loc, double Scale)[n];
        for (int s = 0; s < n; s++)
        {
            var context = contexts[first + s];
            var (loc, scale) = TimeSeriesHistory.InstanceScale(context, ScaleEpsilon);
            scales[s] = (loc, scale);
            int offset = (_patches * _patchSize) - context.Length;   // left padding with missing values
            for (int i = 0; i < context.Length; i++)
            {
                float value = context[i];
                if (float.IsNaN(value))
                {
                    continue;
                }

                int position = offset + i;
                int patch = (s * _patches) + (position / _patchSize);
                int column = position % _patchSize;
                patches[(patch * width) + column] = (float)((value - loc) / scale);
                patches[(patch * width) + _patchSize + column] = 1f;
                mask[patch] = 1f;
            }
        }

        _compiled.SetInput<float>(_patchesInput, patches);
        _compiled.SetInput<float>(_maskInput, mask);
        var run = new RunParameters();
        if (_batched)
        {
            run.Symbols[TimeSeriesGraphs.SeriesSymbol] = n;
        }

        _compiled.Run(run);
        int size = _levels.Length * PredictionLength;
        var output = new float[n * size];
        _compiled.GetOutput<float>(TimeSeriesGraphs.QuantilesOutput, output);
        for (int s = 0; s < n; s++)
        {
            var (loc, scale) = scales[s];
            var result = new float[_levels.Length][];
            for (int q = 0; q < _levels.Length; q++)
            {
                result[q] = new float[PredictionLength];
                for (int t = 0; t < PredictionLength; t++)
                {
                    result[q][t] = (float)((output[(s * size) + (q * PredictionLength) + t] * scale) + loc);
                }
            }

            results[first + s] = result;
        }
    }

    private void Compile(int series)
    {
        _compiled = _batched
            ? _model.CompileComponent(ModelComponentRoles.Encoder, TimeSeriesGraphs.ForecastGraph,
                new Dictionary<string, long> { [TimeSeriesGraphs.SeriesSymbol] = series })
            : _model.CompileComponent(ModelComponentRoles.Encoder, TimeSeriesGraphs.ForecastGraph);
        _patchesInput = _compiled.Graph.Inputs.First(v => v.Input!.Name == TimeSeriesGraphs.PatchesInput);
        _maskInput = _compiled.Graph.Inputs.First(v => v.Input!.Name == TimeSeriesGraphs.MaskInput);
        _seriesBound = series;
    }
}
