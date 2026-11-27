using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// The built-in <see cref="ITimeSeriesForecaster"/> over the patch-decoder form of <see cref="TimeSeriesGraphs"/> (TimesFM). Missing values are filled as the
/// timesfm package does (leading ones dropped, the others linearly interpolated); every pass normalizes the patches of the series (the first-patch statistics of
/// TimesFM 2 or the running statistics of TimesFM 2.5) and forecasts the output patch after the last one; longer horizons append the point forecast and run
/// again. TimesFM 2.5's forecasting flags (<see cref="TimeSeriesGraphs.NormalizeInputsKey"/>, <see cref="TimeSeriesGraphs.FlipInvarianceKey"/>, the continuous
/// quantile head, <see cref="TimeSeriesGraphs.FixQuantileCrossingKey"/>, <see cref="TimeSeriesGraphs.NonNegativeKey"/>) follow the package's
/// <c>compiled_decode</c>.
/// </summary>
internal sealed class PatchDecoderForecasterClient : ITimeSeriesForecaster
{
    private const double Tolerance = 1e-6;
    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _patchesInput;
    private readonly Graph.Value? _positionsInput;
    private readonly Graph.Value _spansInput;
    private readonly Graph.Value _lastRowInput;
    private readonly int _patchSize;
    private readonly int _sequenceLimit;
    private readonly bool _running;
    private readonly int _decodeIndex;
    private readonly int _quantileHorizon;
    private readonly bool _normalizeInputs;
    private readonly bool _flipInvariance;
    private readonly bool _nonNegative;
    private readonly bool _fixQuantileCrossing;
    private readonly float[] _levels;
    private readonly int _columns;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public PatchDecoderForecasterClient(TaskModel model)
    {
        _model = model;
        var component = model.Description.Component(ModelComponentRoles.Decoder);
        ContextLength = component.ConfigValue(TimeSeriesGraphs.ContextLengthKey, 0);
        PredictionLength = component.ConfigValue(TimeSeriesGraphs.PredictionLengthKey, 0);
        _patchSize = component.ConfigValue(TimeSeriesGraphs.PatchSizeKey, 0);
        _sequenceLimit = component.ConfigValue(TimeSeriesGraphs.SequenceLimitKey, ContextLength);
        _levels = TimeSeriesQuantiles.Parse(component.ConfigValue(TimeSeriesGraphs.QuantilesKey, ""));
        _columns = _levels.Length + 1;
        _running = component.ConfigValue(TimeSeriesGraphs.NormalizationKey, TimeSeriesGraphs.FirstPatchNormalization) == TimeSeriesGraphs.RunningNormalization;
        _decodeIndex = component.ConfigValue(TimeSeriesGraphs.DecodeIndexKey, 0);
        _quantileHorizon = component.ConfigValue(TimeSeriesGraphs.QuantileHorizonKey, 0);
        _normalizeInputs = component.ConfigValue(TimeSeriesGraphs.NormalizeInputsKey, false);
        _flipInvariance = component.ConfigValue(TimeSeriesGraphs.FlipInvarianceKey, false);
        _nonNegative = component.ConfigValue(TimeSeriesGraphs.NonNegativeKey, false);
        _fixQuantileCrossing = component.ConfigValue(TimeSeriesGraphs.FixQuantileCrossingKey, false);
        _compiled = model.CompileComponent(ModelComponentRoles.Decoder, TimeSeriesGraphs.PatchDecoderGraph);
        _patchesInput = _compiled.Graph.Inputs.First(v => v.Input!.Name == TimeSeriesGraphs.PatchesInput);
        _positionsInput = _compiled.Graph.Inputs.FirstOrDefault(v => v.Input!.Name == TimeSeriesGraphs.PositionsInput);
        _spansInput = _compiled.Graph.Inputs.First(v => v.Input!.Name == TimeSeriesGraphs.SpansInput);
        _lastRowInput = _compiled.Graph.Inputs.First(v => v.Input!.Name == TimeSeriesGraphs.LastRowInput);
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

    /// <summary>The timesfm package's filling of missing values: leading ones are dropped, the others linearly interpolated between their neighbours.</summary>
    internal static double[] Fill(ReadOnlySpan<float> history)
    {
        int first = 0;
        while (first < history.Length && float.IsNaN(history[first]))
        {
            first++;
        }

        var values = new double[history.Length - first];
        int previous = -1;
        for (int i = 0; i < values.Length; i++)
        {
            float value = history[first + i];
            if (float.IsNaN(value))
            {
                continue;
            }

            values[i] = value;
            for (int j = previous + 1; j < i; j++)
            {
                values[j] = values[previous] + ((values[i] - values[previous]) * (j - previous) / (i - previous));
            }

            previous = i;
        }

        for (int j = previous + 1; j < values.Length; j++)
        {
            values[j] = values[previous];   // trailing gaps keep the last observation, as np.interp does
        }

        return values;
    }

    private TimeSeriesForecast Forecast(float[] history, int horizon, IReadOnlyList<float> levels, CancellationToken cancellationToken)
    {
        var context = Fill(history);
        if (context.Length > ContextLength)
        {
            context = context[^ContextLength..];
        }

        bool positive = _nonNegative && context.All(v => v >= 0);
        double mu = 0;
        double sigma = 1;
        if (_normalizeInputs)
        {
            (mu, sigma) = PaddedStatistics(context);
            double safe = sigma < Tolerance ? 1 : sigma;
            for (int i = 0; i < context.Length; i++)
            {
                context[i] = (context[i] - mu) / safe;
            }
        }

        var (full, spread) = Decode(context, horizon, cancellationToken);
        if (_flipInvariance)
        {
            var (flipped, flippedSpread) = Decode([.. context.Select(v => -v)], horizon, cancellationToken);
            Combine(full, flipped);
            if (spread is not null)
            {
                Combine(spread, flippedSpread!);
            }
        }

        if (spread is not null)
        {
            // The continuous quantile head's spreads around its median, shifted onto the point forecast.
            int d = _decodeIndex;
            for (int t = 0; t < Math.Min(horizon, spread.Length); t++)
            {
                for (int q = 1; q < _columns; q++)
                {
                    if (q != d)
                    {
                        full[t][q] = spread[t][q] - spread[t][d] + full[t][d];
                    }
                }
            }
        }

        var native = new float[_levels.Length][];
        for (int q = 0; q < native.Length; q++)
        {
            native[q] = new float[horizon];
        }

        var point = new float[horizon];
        double scale = sigma < Tolerance ? 1 : sigma;
        for (int t = 0; t < horizon; t++)
        {
            var row = full[t];
            if (_fixQuantileCrossing)
            {
                for (int q = _decodeIndex - 1; q >= 1; q--)
                {
                    row[q] = Math.Min(row[q], row[q + 1]);
                }

                for (int q = _decodeIndex + 1; q < _columns; q++)
                {
                    row[q] = Math.Max(row[q], row[q - 1]);
                }
            }

            for (int c = 0; c < _columns; c++)
            {
                double value = _normalizeInputs ? (row[c] * scale) + mu : row[c];
                row[c] = positive ? Math.Max(value, 0) : value;
            }

            point[t] = (float)row[_decodeIndex];
            for (int q = 0; q < native.Length; q++)
            {
                native[q][t] = (float)row[q + 1];
            }
        }

        return new TimeSeriesForecast(levels, TimeSeriesQuantiles.Select(_levels, native, levels), point);
    }

    // (x − flip(x')) / 2 with flip negating x' and reversing its quantile columns (the point column stays): the flip-invariant forecast.
    private void Combine(double[][] rows, double[][] flipped)
    {
        for (int t = 0; t < rows.Length; t++)
        {
            var row = rows[t];
            var other = flipped[t];
            row[0] = (row[0] - other[0]) / 2;
            var levels = new double[_columns - 1];
            for (int q = 1; q < _columns; q++)
            {
                levels[q - 1] = (row[q] - other[_columns - q]) / 2;
            }

            levels.CopyTo(row, 1);
        }
    }

    // The package's normalize_inputs at max_context = ContextLength: the statistics of the context left-padded with zeros (scaled by the largest magnitude
    // for range), the deviation the sample one.
    private (double Mu, double Sigma) PaddedStatistics(double[] context)
    {
        int count = Math.Max(context.Length, ContextLength);
        double magnitude = Math.Max(1, context.Max(Math.Abs));
        double sum = 0;
        foreach (double v in context)
        {
            sum += v / magnitude;
        }

        double mean = sum / count;
        double squares = (count - context.Length) * mean * mean;
        foreach (double v in context)
        {
            double d = (v / magnitude) - mean;
            squares += d * d;
        }

        double deviation = count > 1 ? Math.Sqrt(squares / (count - 1)) : 0;
        return (mean * magnitude, deviation * magnitude);
    }

    // The forecast rows of whole output patches covering the horizon (columns: point, levels), and the continuous quantile head of the first pass.
    private (double[][] Rows, double[][]? Spread) Decode(double[] context, int horizon, CancellationToken cancellationToken)
    {
        var sequence = new List<double>(context);
        var rows = new List<double[]>();
        double[][]? spread = null;
        while (rows.Count < horizon)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int start = Math.Max(0, sequence.Count - _sequenceLimit);
            var (point, quantiles) = RunPass(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(sequence)[start..], spread is null && _quantileHorizon > 0);
            spread ??= quantiles;
            rows.AddRange(point);
            foreach (var row in point)
            {
                sequence.Add(row[_decodeIndex]);
            }
        }

        return (rows.ToArray(), spread);
    }

    // One pass over a gap-free series: the forecast of the output patch after it in the series' units, [PredictionLength][columns], and optionally the
    // continuous quantile head [quantile horizon][columns].
    private (double[][] Point, double[][]? Spread) RunPass(ReadOnlySpan<double> series, bool withSpread)
    {
        int ps = _patchSize;
        int patches = (series.Length + ps - 1) / ps;
        int offset = (patches * ps) - series.Length;   // left padding of the first patch
        var mu = new double[patches];
        var sigma = new double[patches];
        if (_running)
        {
            RunningStatistics(series, offset, mu, sigma);
        }
        else
        {
            var (m, s) = FirstPatchStatistics(series, offset, patches);
            Array.Fill(mu, m);
            Array.Fill(sigma, s);
        }

        var input = new float[patches * 2 * ps];
        for (int i = 0; i < series.Length; i++)
        {
            int position = offset + i;
            int patch = position / ps;
            double safe = sigma[patch] < Tolerance ? 1 : sigma[patch];
            input[(patch * 2 * ps) + (position % ps)] = (float)((series[i] - mu[patch]) / safe);
            input[(patch * 2 * ps) + ps + (position % ps)] = 1f;
        }

        _compiled.SetInput<float>(_patchesInput, input);
        if (_positionsInput is not null)
        {
            _compiled.SetInput<int>(_positionsInput, [.. Enumerable.Range(0, patches)]);
        }

        var spans = new int[2 * patches];
        for (int row = 0; row < patches; row++)
        {
            spans[(2 * row) + 1] = patches;
        }

        _compiled.SetInput<int>(_spansInput, spans);
        _compiled.SetInput<int>(_lastRowInput, [patches - 1]);
        var run = new RunParameters();
        run.Symbols[TimeSeriesGraphs.PatchSymbol] = patches;
        if (!withSpread && _quantileHorizon > 0)
        {
            run.SkipOutputs.Add(TimeSeriesGraphs.QuantileSpreadOutput);
        }

        _compiled.Run(run);
        double loc = mu[^1];
        double scale = sigma[^1] < Tolerance ? 1 : sigma[^1];
        var point = Read(TimeSeriesGraphs.PointOutput, PredictionLength, loc, scale);
        return (point, withSpread ? Read(TimeSeriesGraphs.QuantileSpreadOutput, _quantileHorizon, loc, scale) : null);
    }

    private double[][] Read(string output, int steps, double loc, double scale)
    {
        var values = new float[steps * _columns];
        _compiled.GetOutput<float>(output, values);
        var rows = new double[steps][];
        for (int t = 0; t < steps; t++)
        {
            rows[t] = new double[_columns];
            for (int c = 0; c < _columns; c++)
            {
                rows[t][c] = (values[(t * _columns) + c] * scale) + loc;
            }
        }

        return rows;
    }

    // TimesFM 2: the mean and (population) deviation of the first patch with at least three observed values, the last patch when there is none; the deviation at least 1e-6.
    private (double Mu, double Sigma) FirstPatchStatistics(ReadOnlySpan<double> series, int offset, int patches)
    {
        int ps = _patchSize;
        int chosen = patches - 1;
        for (int patch = 0; patch < patches; patch++)
        {
            int valid = Math.Min(ps, ((patch + 1) * ps) - offset) - Math.Max(0, (patch * ps) - offset);
            if (valid >= 3)
            {
                chosen = patch;
                break;
            }
        }

        int from = Math.Max(0, (chosen * ps) - offset);
        int to = ((chosen + 1) * ps) - offset;
        var values = series[from..to];
        double mean = 0;
        foreach (double v in values)
        {
            mean += v;
        }

        mean /= Math.Max(1, values.Length);
        double variance = 0;
        foreach (double v in values)
        {
            variance += (v - mean) * (v - mean);
        }

        variance /= Math.Max(1, values.Length);
        return (mean, Math.Max(Math.Sqrt(variance), Tolerance));
    }

    // TimesFM 2.5: per patch the mean and (population) deviation of all observed values up to and including it.
    private void RunningStatistics(ReadOnlySpan<double> series, int offset, double[] mu, double[] sigma)
    {
        int ps = _patchSize;
        double count = 0;
        double mean = 0;
        double m2 = 0;
        for (int patch = 0; patch < mu.Length; patch++)
        {
            int from = Math.Max(0, (patch * ps) - offset);
            int to = ((patch + 1) * ps) - offset;
            foreach (double v in series[from..to])
            {
                count++;
                double delta = v - mean;
                mean += delta / count;
                m2 += delta * (v - mean);
            }

            mu[patch] = mean;
            sigma[patch] = Math.Sqrt(Math.Max(m2 / count, 0));
        }
    }
}
