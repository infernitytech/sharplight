using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// The built-in <see cref="ITimeSeriesForecaster"/> over the quantile-decoder form of <see cref="TimeSeriesGraphs"/> (Moirai 2.0), following uni2ts'
/// <c>Moirai2Forecast</c>: the last row of the context predicts <see cref="PredictionLength"/> steps at every level. Longer horizons extend one sequence
/// per level with that level's forecast (each re-scaled with its own statistics, which include the fed-back values), run them packed in one pass, and take
/// the levels of all levels × levels forecasts of every step as the next block.
/// </summary>
internal sealed class QuantileDecoderForecasterClient : ITimeSeriesForecaster
{
    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _patchesInput;
    private readonly Graph.Value _positionsInput;
    private readonly Graph.Value _spansInput;
    private readonly Graph.Value _readInput;
    private readonly int _patchSize;
    private readonly int _tokens;
    private readonly int _maxHorizon;
    private readonly float[] _levels;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public QuantileDecoderForecasterClient(TaskModel model)
    {
        _model = model;
        var component = model.Description.Component(ModelComponentRoles.Decoder);
        ContextLength = component.ConfigValue(TimeSeriesGraphs.ContextLengthKey, 0);
        _patchSize = component.ConfigValue(TimeSeriesGraphs.PatchSizeKey, 0);
        _tokens = component.ConfigValue(TimeSeriesGraphs.PredictTokensKey, 1);
        PredictionLength = component.ConfigValue(TimeSeriesGraphs.PredictionLengthKey, _tokens * _patchSize);
        _maxHorizon = component.ConfigValue(TimeSeriesGraphs.MaxHorizonKey, 2048);
        _levels = TimeSeriesQuantiles.Parse(component.ConfigValue(TimeSeriesGraphs.QuantilesKey, ""));
        _compiled = model.CompileComponent(ModelComponentRoles.Decoder, TimeSeriesGraphs.QuantileDecoderGraph);
        _patchesInput = Input(TimeSeriesGraphs.PatchesInput);
        _positionsInput = Input(TimeSeriesGraphs.PositionsInput);
        _spansInput = Input(TimeSeriesGraphs.SpansInput);
        _readInput = Input(TimeSeriesGraphs.ReadRowsInput);
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
        if (horizon > _maxHorizon)
        {
            throw new ArgumentOutOfRangeException(nameof(options), $"The horizon {horizon} exceeds the model's longest horizon {_maxHorizon}.");
        }

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

    private Graph.Value Input(string name) => _compiled.Graph.Inputs.First(v => v.Input!.Name == name);

    private TimeSeriesForecast Forecast(float[] history, int horizon, IReadOnlyList<float> levels, CancellationToken cancellationToken)
    {
        int ps = _patchSize;
        int count = _levels.Length;
        var window = history.Length <= ContextLength ? history : history[^ContextLength..];
        var (values, observed) = MoiraiPatches.Pad(window, ps);
        var times = MoiraiPatches.TimeIds(observed, ps);
        int patches = (horizon + ps - 1) / ps;
        var native = new float[count][];
        for (int q = 0; q < count; q++)
        {
            native[q] = new float[patches * ps];
        }

        // The first pass: the context alone; its last row predicts the first tokens at every level.
        var first = Run(values, observed, times, [native[0]], 0, cancellationToken)[0];
        int filled = Math.Min(_tokens, patches);
        for (int j = 0; j < filled; j++)
        {
            for (int q = 0; q < count; q++)
            {
                first.AsSpan(((j * count) + q) * ps, ps).CopyTo(native[q].AsSpan(j * ps));
            }
        }

        var pool = new float[count * count];
        while (filled < patches)
        {
            int take = Math.Min(_tokens, patches - filled);
            var outputs = Run(values, observed, times, native, filled, cancellationToken);
            for (int j = 0; j < take; j++)
            {
                for (int column = 0; column < ps; column++)
                {
                    for (int s = 0; s < count; s++)
                    {
                        for (int q = 0; q < count; q++)
                        {
                            pool[(s * count) + q] = outputs[s][(((j * count) + q) * ps) + column];
                        }
                    }

                    Array.Sort(pool);
                    for (int q = 0; q < count; q++)
                    {
                        native[q][((filled + j) * ps) + column] = TimeSeriesQuantiles.Of(pool, _levels[q]);
                    }
                }
            }

            filled += take;
        }

        var forecasts = native.Select(row => row[..horizon]).ToArray();
        int median = TimeSeriesQuantiles.IndexOf(_levels, 0.5f);
        var mean = median >= 0 ? (float[])forecasts[median].Clone() : TimeSeriesQuantiles.Select(_levels, forecasts, [0.5f])[0];
        return new TimeSeriesForecast(levels, TimeSeriesQuantiles.Select(_levels, forecasts, levels), mean);
    }

    // One pass over the sequences "context + the first `filled` forecast patches of sequence s" (one sequence per entry of `sequences`), packed:
    // per sequence the de-scaled prediction of its last row, [tokens][levels][patch size].
    private float[][] Run(float[] values, bool[] observed, int[] times, float[][] sequences, int filled, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int ps = _patchSize;
        int context = times.Length;
        int length = context + filled;
        int rows = sequences.Length * length;
        int width = 2 * ps;
        var patches = new float[rows * width];
        var positions = new int[rows];
        var spans = new int[rows * 2];
        var read = new int[sequences.Length];
        var scales = new (float Loc, float Scale)[sequences.Length];
        var sequenceValues = new float[length * ps];
        var sequenceObserved = new bool[length * ps];
        values.CopyTo(sequenceValues, 0);
        observed.CopyTo(sequenceObserved, 0);
        sequenceObserved.AsSpan(values.Length).Fill(true);
        var sequenceTimes = new int[length];
        times.CopyTo(sequenceTimes, 0);
        for (int j = 0; j < filled; j++)
        {
            sequenceTimes[context + j] = times[^1] + 1 + j;
        }

        for (int s = 0; s < sequences.Length; s++)
        {
            sequences[s].AsSpan(0, filled * ps).CopyTo(sequenceValues.AsSpan(values.Length));
            var (loc, scale) = MoiraiPatches.Scale(sequenceValues, sequenceObserved);
            scales[s] = (loc, scale);
            int start = s * length;
            read[s] = start + length - 1;
            for (int r = 0; r < length; r++)
            {
                int row = start + r;
                for (int c = 0; c < ps; c++)
                {
                    int i = (r * ps) + c;
                    patches[(row * width) + c] = MoiraiPatches.Scaled(sequenceValues[i], loc, scale);
                    patches[(row * width) + ps + c] = sequenceObserved[i] ? 1f : 0f;
                }

                // Causal in time: the rows whose time id is at most this row's (leading unobserved patches share id 0 and see each other).
                int end = r;
                while (end + 1 < length && sequenceTimes[end + 1] == sequenceTimes[r])
                {
                    end++;
                }

                positions[row] = sequenceTimes[r];
                spans[2 * row] = start;
                spans[(2 * row) + 1] = end + 1;
            }
        }

        _compiled.SetInput<float>(_patchesInput, patches);
        _compiled.SetInput<int>(_positionsInput, positions);
        _compiled.SetInput<int>(_spansInput, spans);
        _compiled.SetInput<int>(_readInput, read);
        var run = new RunParameters();
        run.Symbols[TimeSeriesGraphs.PatchSymbol] = rows;
        run.Symbols[TimeSeriesGraphs.ReadSymbol] = sequences.Length;
        _compiled.Run(run);
        int size = _tokens * _levels.Length * ps;
        var output = new float[sequences.Length * size];
        _compiled.GetOutput<float>(TimeSeriesGraphs.TokenQuantilesOutput, output);
        var result = new float[sequences.Length][];
        for (int s = 0; s < sequences.Length; s++)
        {
            result[s] = new float[size];
            for (int i = 0; i < size; i++)
            {
                result[s][i] = (output[(s * size) + i] * scales[s].Scale) + scales[s].Loc;
            }
        }

        return result;
    }
}
