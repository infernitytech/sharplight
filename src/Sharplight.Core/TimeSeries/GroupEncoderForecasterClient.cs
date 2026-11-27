using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// The built-in <see cref="ITimeSeriesForecaster"/> over the group-encoder form of <see cref="TimeSeriesGraphs"/> (Chronos-2), following chronos-forecasting's
/// <c>Chronos2Pipeline</c>: the targets and covariates of a request form one group (targets first, then past-only, then known-future covariates), one pass
/// predicts up to <see cref="PredictionLength"/> steps of every series, and longer horizons unroll: every series' context is extended by its forecast at each of
/// the levels 0.1 … 0.9 (one group per level; known future covariate values replace their forecasts), the groups run packed in one pass, and the weighted
/// quantiles of all level-by-level forecasts give the next block.
/// </summary>
internal sealed class GroupEncoderForecasterClient : ITimeSeriesForecaster
{
    private const float ScaleEpsilon = 1e-5f;
    private const float MaskedScore = -1e9f;
    private static readonly float[] s_unrolledLevels = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
    private readonly TaskModel _model;
    private readonly int _patchSize;
    private readonly int _maxOutputPatches;
    private readonly float _timeScale;
    private readonly bool _arcsinh;
    private readonly bool _register;
    private readonly float[] _levels;
    private readonly long _defaultRows;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private CompiledModel? _compiled;
    private long _rowBound;
    private long _readBound;
    private bool _disposed;

    public GroupEncoderForecasterClient(TaskModel model)
    {
        _model = model;
        var component = model.Description.Component(ModelComponentRoles.Encoder);
        ContextLength = component.ConfigValue(TimeSeriesGraphs.ContextLengthKey, 0);
        _patchSize = component.ConfigValue(TimeSeriesGraphs.PatchSizeKey, 0);
        _maxOutputPatches = component.ConfigValue(TimeSeriesGraphs.MaxOutputPatchesKey, 1);
        PredictionLength = component.ConfigValue(TimeSeriesGraphs.PredictionLengthKey, _maxOutputPatches * _patchSize);
        _timeScale = component.ConfigValue(TimeSeriesGraphs.TimeEncodingScaleKey, ContextLength);
        _arcsinh = component.ConfigValue(TimeSeriesGraphs.ArcsinhKey, false);
        _register = component.ConfigValue(TimeSeriesGraphs.RegisterTokenKey, true);
        _levels = TimeSeriesQuantiles.Parse(component.ConfigValue(TimeSeriesGraphs.QuantilesKey, ""));
        _defaultRows = component.SymbolBounds.GetValueOrDefault(TimeSeriesGraphs.PatchSymbol, 1);
        Compile(_defaultRows, component.SymbolBounds.GetValueOrDefault(TimeSeriesGraphs.ReadSymbol, _maxOutputPatches));
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int ContextLength { get; }

    public int PredictionLength { get; }

    public IReadOnlyList<float> QuantileLevels => _levels;

    public bool SupportsCovariates => true;

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public async Task<TimeSeriesForecast> ForecastAsync(ReadOnlyMemory<float> history, TimeSeriesForecastOptions? options = null, CancellationToken cancellationToken = default)
    {
        TimeSeriesHistory.Validate(history.Span);
        var forecast = await ForecastAsync(TimeSeriesForecastRequest.Univariate(history), options, cancellationToken).ConfigureAwait(false);
        return forecast.Targets[0];
    }

    public Task<TimeSeriesMultivariateForecast> ForecastAsync(TimeSeriesForecastRequest request, TimeSeriesForecastOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(request);
        int horizon = TimeSeriesHistory.Horizon(request, options, PredictionLength);
        TimeSeriesHistory.Validate(request, horizon);
        var levels = TimeSeriesQuantiles.Requested(options?.QuantileLevels, _levels);

        // The group's rows as the reference orders them: targets, past-only covariates, known-future covariates.
        var contexts = new List<float[]>();
        var futures = new List<float[]?>();
        foreach (var target in request.Targets)
        {
            contexts.Add(target.ToArray());
            futures.Add(null);
        }

        foreach (var covariate in request.Covariates.Where(c => c.Future.IsEmpty).Concat(request.Covariates.Where(c => !c.Future.IsEmpty)))
        {
            contexts.Add(covariate.History.ToArray());
            futures.Add(covariate.Future.IsEmpty ? null : covariate.Future.ToArray());
        }

        int targets = request.Targets.Count;
        return BackgroundCompute.RunAsync(_gate, () => Forecast([.. contexts], [.. futures], targets, horizon, levels, cancellationToken), cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _compiled?.Dispose();
        _gate.Dispose();
    }

    private TimeSeriesMultivariateForecast Forecast(float[][] contexts, float[]?[] futures, int targets, int horizon, IReadOnlyList<float> levels,
        CancellationToken cancellationToken)
    {
        int series = contexts.Length;
        int count = _levels.Length;
        var native = new float[targets][][];
        for (int s = 0; s < targets; s++)
        {
            native[s] = new float[count][];
            for (int q = 0; q < count; q++)
            {
                native[s][q] = new float[horizon];
            }
        }

        // prediction[s][q][step] of the last block; the first pass is one group of every series.
        int done = 0;
        int patches = OutputPatches(horizon);
        var prediction = Pass([contexts.Select(c => Tail(c, ContextLength)).ToArray()], futures, 0, patches)[0];
        Store(prediction, native, done, horizon);
        done += patches * _patchSize;
        if (done >= horizon)
        {
            return Result(native, levels);
        }

        // Long horizons: one path per unrolled level, each its own group; the samples of a step are paths × levels, weighted by their probability mass.
        var unrolled = s_unrolledLevels.Where(level => TimeSeriesQuantiles.IndexOf(_levels, level) >= 0).ToArray();
        var pathMass = TimeSeriesQuantiles.ProbabilityMass(unrolled);
        var levelMass = TimeSeriesQuantiles.ProbabilityMass(_levels);
        var paths = new float[unrolled.Length][][];
        for (int u = 0; u < paths.Length; u++)
        {
            paths[u] = [.. contexts];
        }

        var column = new float[count];
        var samples = new float[unrolled.Length * count];
        var weights = new float[samples.Length];
        var reduced = new float[count];
        while (done < horizon)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int length = prediction[0][0].Length;
            for (int u = 0; u < paths.Length; u++)
            {
                for (int s = 0; s < series; s++)
                {
                    var extended = new float[paths[u][s].Length + length];
                    paths[u][s].CopyTo(extended, 0);
                    for (int t = 0; t < length; t++)
                    {
                        for (int q = 0; q < count; q++)
                        {
                            column[q] = prediction[s][q][t];
                        }

                        // A known future covariate value replaces the forecast (the reference's _slide_context_and_future_covariates).
                        float known = futures[s] is { } future && done - length + t < future.Length ? future[done - length + t] : float.NaN;
                        extended[paths[u][s].Length + t] = float.IsNaN(known) ? TimeSeriesQuantiles.Interpolate(_levels, column, unrolled[u]) : known;
                    }

                    paths[u][s] = Tail(extended, ContextLength);
                }
            }

            patches = OutputPatches(horizon - done);
            var passes = Pass(paths, futures, done, patches);
            int steps = patches * _patchSize;
            prediction = new float[series][][];
            for (int s = 0; s < series; s++)
            {
                prediction[s] = new float[count][];
                for (int q = 0; q < count; q++)
                {
                    prediction[s][q] = new float[steps];
                }

                for (int t = 0; t < steps; t++)
                {
                    for (int u = 0; u < paths.Length; u++)
                    {
                        for (int q = 0; q < count; q++)
                        {
                            samples[(u * count) + q] = passes[u][s][q][t];
                            weights[(u * count) + q] = pathMass[u] * levelMass[q];
                        }
                    }

                    TimeSeriesQuantiles.Weighted(samples, weights, _levels, reduced);
                    for (int q = 0; q < count; q++)
                    {
                        prediction[s][q][t] = reduced[q];
                    }
                }
            }

            Store(prediction, native, done, horizon);
            done += steps;
        }

        return Result(native, levels);
    }

    private int OutputPatches(int remaining) => Math.Min((remaining + _patchSize - 1) / _patchSize, _maxOutputPatches);

    // Copies the target rows of a block into the native forecast from step `offset` on.
    private static void Store(float[][][] block, float[][][] native, int offset, int horizon)
    {
        for (int s = 0; s < native.Length; s++)
        {
            for (int q = 0; q < native[s].Length; q++)
            {
                int steps = Math.Min(block[s][q].Length, horizon - offset);
                Array.Copy(block[s][q], 0, native[s][q], offset, steps);
            }
        }
    }

    private TimeSeriesMultivariateForecast Result(float[][][] native, IReadOnlyList<float> levels)
    {
        int median = TimeSeriesQuantiles.IndexOf(_levels, 0.5f);
        var forecasts = new TimeSeriesForecast[native.Length];
        for (int s = 0; s < native.Length; s++)
        {
            forecasts[s] = new TimeSeriesForecast(levels, Select(native[s], levels), median >= 0 ? (float[])native[s][median].Clone() : Select(native[s], [0.5f])[0]);
        }

        return new TimeSeriesMultivariateForecast(forecasts);
    }

    // The requested levels: the native ones as they are, others interpolated as Chronos2Pipeline.predict_quantiles does.
    private float[][] Select(float[][] native, IReadOnlyList<float> levels)
    {
        int horizon = native[0].Length;
        var result = new float[levels.Count][];
        bool subset = levels.All(level => TimeSeriesQuantiles.IndexOf(_levels, level) >= 0);
        var column = new float[_levels.Length];
        for (int l = 0; l < levels.Count; l++)
        {
            result[l] = subset ? (float[])native[TimeSeriesQuantiles.IndexOf(_levels, levels[l])].Clone() : new float[horizon];
        }

        if (subset)
        {
            return result;
        }

        for (int t = 0; t < horizon; t++)
        {
            for (int q = 0; q < _levels.Length; q++)
            {
                column[q] = native[q][t];
            }

            for (int l = 0; l < levels.Count; l++)
            {
                result[l][t] = TimeSeriesQuantiles.Interpolate(_levels, column, levels[l]);
            }
        }

        return result;
    }

    private static float[] Tail(float[] values, int length) => values.Length <= length ? values : values[^length..];

    private float Transform(float value, double loc, double scale)
    {
        float scaled = (float)((value - loc) / scale);
        return _arcsinh ? MathF.Asinh(scaled) : scaled;
    }

    // One model pass over groups of series (every group the same series, each context at most ContextLength long): per group, series and level the unscaled
    // forecast of `patches` horizon patches. `offset` is the first horizon step of the pass (for the known future covariate values).
    private float[][][][] Pass(float[][][] groups, float[]?[] futures, int offset, int patches)
    {
        int p = _patchSize;
        int seriesCount = groups[0].Length;
        int length = groups[0].Max(c => c.Length);
        int contextPatches = (length + p - 1) / p;
        int sequence = contextPatches + (_register ? 1 : 0) + patches;
        int rows = groups.Length * seriesCount * sequence;
        int reads = groups.Length * seriesCount * patches;
        EnsureCompiled(rows, reads);
        var compiled = _compiled!;
        int width = 3 * p;
        var input = new float[rows * width];
        var register = new float[rows];
        var bias = new float[rows];
        var positions = new int[rows];
        var spans = new int[rows * 2];
        var order = new int[rows];
        var inverse = new int[rows];
        var groupSpans = new int[rows * 2];
        var read = new int[reads];
        var scales = new (double Loc, double Scale)[groups.Length * seriesCount];
        for (int g = 0; g < groups.Length; g++)
        {
            for (int n = 0; n < seriesCount; n++)
            {
                int index = (g * seriesCount) + n;
                var context = groups[g][n];
                var (loc, scale) = scales[index] = TimeSeriesHistory.InstanceScale(context, ScaleEpsilon);
                int first = index * sequence;
                int pad = (contextPatches * p) - context.Length;   // left padding with missing values (the reference pads to the batch's longest)
                for (int j = 0; j < sequence; j++)
                {
                    int row = first + j;
                    positions[row] = j;
                    spans[2 * row] = first;
                    spans[(2 * row) + 1] = sequence;
                }

                for (int j = 0; j < contextPatches; j++)
                {
                    int row = first + j;
                    bool observed = false;
                    for (int i = 0; i < p; i++)
                    {
                        int position = (j * p) + i;
                        input[(row * width) + i] = (float)(position - (contextPatches * p)) / _timeScale;
                        int at = position - pad;
                        if (at >= 0 && !float.IsNaN(context[at]))
                        {
                            input[(row * width) + p + i] = Transform(context[at], loc, scale);
                            input[(row * width) + (2 * p) + i] = 1f;
                            observed = true;
                        }
                    }

                    bias[row] = observed ? 0f : MaskedScore;
                }

                if (_register)
                {
                    register[first + contextPatches] = 1f;
                }

                int future = first + contextPatches + (_register ? 1 : 0);
                for (int f = 0; f < patches; f++)
                {
                    int row = future + f;
                    read[(index * patches) + f] = row;
                    for (int i = 0; i < p; i++)
                    {
                        int step = (f * p) + i;
                        input[(row * width) + i] = (float)step / _timeScale;
                        if (futures[n] is { } known && offset + step < known.Length && !float.IsNaN(known[offset + step]))
                        {
                            input[(row * width) + p + i] = Transform(known[offset + step], loc, scale);
                            input[(row * width) + (2 * p) + i] = 1f;
                        }
                    }
                }
            }
        }

        // The group order: group by group, position by position, series by series; every row sees its group's rows at its position.
        int k = 0;
        for (int g = 0; g < groups.Length; g++)
        {
            for (int j = 0; j < sequence; j++)
            {
                int start = k;
                for (int n = 0; n < seriesCount; n++)
                {
                    int row = (((g * seriesCount) + n) * sequence) + j;
                    order[k] = row;
                    inverse[row] = k;
                    groupSpans[2 * k] = start;
                    groupSpans[(2 * k) + 1] = seriesCount;
                    k++;
                }
            }
        }

        compiled.SetInput<float>(Input(TimeSeriesGraphs.PatchesInput), input);
        compiled.SetInput<float>(Input(TimeSeriesGraphs.RegisterRowsInput), register);
        compiled.SetInput<float>(Input(TimeSeriesGraphs.KeyBiasInput), bias);
        compiled.SetInput<int>(Input(TimeSeriesGraphs.PositionsInput), positions);
        compiled.SetInput<int>(Input(TimeSeriesGraphs.SpansInput), spans);
        compiled.SetInput<int>(Input(TimeSeriesGraphs.GroupOrderInput), order);
        compiled.SetInput<int>(Input(TimeSeriesGraphs.GroupInverseInput), inverse);
        compiled.SetInput<int>(Input(TimeSeriesGraphs.GroupSpansInput), groupSpans);
        compiled.SetInput<int>(Input(TimeSeriesGraphs.ReadRowsInput), read);
        var run = new RunParameters();
        run.Symbols[TimeSeriesGraphs.PatchSymbol] = rows;
        run.Symbols[TimeSeriesGraphs.ReadSymbol] = reads;
        compiled.Run(run);
        int levels = _levels.Length;
        int size = levels * p;
        var output = new float[reads * size];
        compiled.GetOutput<float>(TimeSeriesGraphs.PatchQuantilesOutput, output);

        var result = new float[groups.Length][][][];
        for (int g = 0; g < groups.Length; g++)
        {
            result[g] = new float[seriesCount][][];
            for (int n = 0; n < seriesCount; n++)
            {
                int index = (g * seriesCount) + n;
                var (loc, scale) = scales[index];
                result[g][n] = new float[levels][];
                for (int q = 0; q < levels; q++)
                {
                    var values = result[g][n][q] = new float[patches * p];
                    for (int f = 0; f < patches; f++)
                    {
                        for (int i = 0; i < p; i++)
                        {
                            float v = output[(((index * patches) + f) * size) + (q * p) + i];
                            values[(f * p) + i] = (float)(((_arcsinh ? MathF.Sinh(v) : v) * scale) + loc);
                        }
                    }
                }
            }
        }

        return result;
    }

    private Graph.Value Input(string name) => _compiled!.Graph.Inputs.First(v => v.Input!.Name == name);

    // Compiles for at least `rows` patch rows and `reads` read rows; a larger group than any before recompiles with the bounds doubled up to it.
    private void EnsureCompiled(long rows, long reads)
    {
        if (rows <= _rowBound && reads <= _readBound)
        {
            return;
        }

        long rowBound = _rowBound;
        while (rowBound < rows)
        {
            rowBound *= 2;
        }

        long readBound = _readBound;
        while (readBound < reads)
        {
            readBound *= 2;
        }

        _compiled?.Dispose();
        _compiled = null;
        Compile(rowBound, readBound);
    }

    private void Compile(long rows, long reads)
    {
        _compiled = _model.CompileComponent(ModelComponentRoles.Encoder, TimeSeriesGraphs.GroupEncoderGraph, new Dictionary<string, long>
        {
            [TimeSeriesGraphs.PatchSymbol] = Math.Max(rows, _defaultRows),
            [TimeSeriesGraphs.ReadSymbol] = reads,
        });
        _rowBound = Math.Max(rows, _defaultRows);
        _readBound = reads;
    }
}
