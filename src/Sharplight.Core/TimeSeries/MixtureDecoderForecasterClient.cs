using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// The built-in <see cref="ITimeSeriesForecaster"/> over the mixture-decoder form of <see cref="TimeSeriesGraphs"/> (Moirai-MoE), following uni2ts'
/// <c>MoiraiMoEForecast</c>: the context patches run causally in time, and the last row's mixture is the next patch. Every one of the
/// <see cref="TimeSeriesForecastOptions.SampleCount"/> sample paths draws that patch, appends it to its own context (re-scaled with the drawn values,
/// as the module's scaler counts them as observed) and predicts the next, until the horizon is covered; up to <see cref="MaxPathsPerPass"/> paths
/// run packed in one pass. The quantiles are the paths' empirical quantiles, the mean their mean. Missing values are 0 with a false observation flag.
/// </summary>
internal sealed class MixtureDecoderForecasterClient : ITimeSeriesForecaster
{
    /// <summary>The sample paths one pass runs (the graph's row bound is this many full sequences).</summary>
    public const int MaxPathsPerPass = 100;

    private static readonly float[] s_levels = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
    private readonly CompiledModel _compiled;
    private readonly TaskModel _model;
    private readonly int _patchSize;
    private readonly int _maxHorizon;
    private readonly int _defaultSamples;
    private readonly MixtureDistribution _mixture;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public MixtureDecoderForecasterClient(TaskModel model)
    {
        _model = model;
        var component = model.Description.Component(ModelComponentRoles.Decoder);
        ContextLength = component.ConfigValue(TimeSeriesGraphs.ContextLengthKey, 0);
        PredictionLength = component.ConfigValue(TimeSeriesGraphs.PredictionLengthKey, 64);
        _patchSize = component.ConfigValue(TimeSeriesGraphs.PatchSizeKey, 16);
        _maxHorizon = component.ConfigValue(TimeSeriesGraphs.MaxHorizonKey, 2048);
        _defaultSamples = component.ConfigValue(TimeSeriesGraphs.SampleCountKey, 100);
        _mixture = new MixtureDistribution(component.ConfigValue(TimeSeriesGraphs.ComponentsKey, ""));
        _compiled = model.CompileComponent(ModelComponentRoles.Decoder, TimeSeriesGraphs.MixtureDecoderGraph);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int ContextLength { get; }

    public int PredictionLength { get; }

    public IReadOnlyList<float> QuantileLevels => s_levels;

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<TimeSeriesForecast> ForecastAsync(ReadOnlyMemory<float> history, TimeSeriesForecastOptions? options = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        TimeSeriesHistory.Validate(history.Span);
        int horizon = CheckHorizon(options?.Horizon ?? PredictionLength);
        int samples = options?.SampleCount ?? _defaultSamples;
        ArgumentOutOfRangeException.ThrowIfLessThan(samples, 1, nameof(options));
        if (options?.PatchSize is int size && size != _patchSize && size != TimeSeriesForecastOptions.AutoPatchSize)
        {
            throw new ArgumentOutOfRangeException(nameof(options), $"Patch size {size} is not the model's ({_patchSize}).");
        }

        var levels = TimeSeriesQuantiles.Requested(options?.QuantileLevels, s_levels);
        var random = options?.Seed is { } seed ? new Random(seed) : new Random();
        var context = history.ToArray();
        return BackgroundCompute.RunAsync(_gate, () => Forecast(context, horizon, samples, new VariateSampler(random), levels, cancellationToken), cancellationToken);
    }

    /// <summary>
    /// The predictive distribution of the patch after the context and the fed patches <paramref name="fed"/> (whole patches, in the series' scale; for
    /// parity tests): that pass's scaler loc and scale and per step of the patch the mapped mixture parameters (layout of <see cref="MixtureDistribution.Map"/>).
    /// </summary>
    internal Task<(float Loc, float Scale, double[][] Parameters)> StepDistributionAsync(float[] history, float[] fed)
    {
        TimeSeriesHistory.Validate(history);
        if (fed.Length % _patchSize != 0)
        {
            throw new ArgumentException($"The fed values must be whole patches of {_patchSize}.", nameof(fed));
        }

        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var context = new PaddedContext(Window(history), _patchSize);
            var (scales, parameters) = Pass(context, [fed], fed.Length / _patchSize);
            return (scales[0].Loc, scales[0].Scale, parameters[0]);
        }, CancellationToken.None);
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

    private float[] Window(float[] history) => history.Length <= ContextLength ? history : history[^ContextLength..];

    private int CheckHorizon(int horizon)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(horizon, 1, "options");
        if (horizon > _maxHorizon)
        {
            throw new ArgumentOutOfRangeException("options", $"The horizon {horizon} exceeds the model's longest horizon {_maxHorizon}.");
        }

        return horizon;
    }

    private TimeSeriesForecast Forecast(float[] history, int horizon, int sampleCount, VariateSampler sampler, IReadOnlyList<float> levels,
        CancellationToken cancellationToken)
    {
        int ps = _patchSize;
        int patches = (horizon + ps - 1) / ps;
        var context = new PaddedContext(Window(history), ps);
        var paths = new float[sampleCount][];
        for (int s = 0; s < sampleCount; s++)
        {
            paths[s] = new float[patches * ps];
        }

        for (int j = 0; j < patches; j++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (j == 0)
            {
                // Every path starts from the same context: one pass, sampled per path.
                var (scales, parameters) = Pass(context, [paths[0]], 0);
                for (int s = 0; s < sampleCount; s++)
                {
                    Draw(paths[s], 0, scales[0], parameters[0], sampler);
                }

                continue;
            }

            for (int first = 0; first < sampleCount; first += MaxPathsPerPass)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var group = paths[first..Math.Min(sampleCount, first + MaxPathsPerPass)];
                var (scales, parameters) = Pass(context, group, j);
                for (int s = 0; s < group.Length; s++)
                {
                    Draw(group[s], j, scales[s], parameters[s], sampler);
                }
            }
        }

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
                column[s] = paths[s][t];
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

    // Draws patch j of a path: uni2ts' AffineTransformed sample, the scaled draw · scale + loc, in float.
    private void Draw(float[] path, int j, (float Loc, float Scale) scale, double[][] parameters, VariateSampler sampler)
    {
        for (int c = 0; c < _patchSize; c++)
        {
            path[(j * _patchSize) + c] = ((float)_mixture.Sample(parameters[c], sampler) * scale.Scale) + scale.Loc;
        }
    }

    /// <summary>
    /// One pass over the paths (packed, each its own causal block): the context patches, then the path's first <paramref name="fed"/> patches as
    /// observed values. Returns per path its scaler loc and scale and the mapped mixture parameters of the next patch's steps.
    /// </summary>
    private ((float Loc, float Scale)[] Scales, double[][][] Parameters) Pass(PaddedContext context, float[][] paths, int fed)
    {
        int ps = _patchSize;
        int perPath = context.Rows + fed;
        int rows = paths.Length * perPath;
        var patches = new float[rows * ps];
        var positions = new int[rows];
        var spans = new int[rows * 2];
        var read = new int[paths.Length];
        var scales = new (float Loc, float Scale)[paths.Length];
        int lastTime = context.Rows == 0 ? 0 : context.Times[^1];
        var values = new float[context.Values.Length + (fed * ps)];
        var observed = new bool[values.Length];
        context.Values.CopyTo(values, 0);
        context.Observed.CopyTo(observed, 0);
        observed.AsSpan(context.Values.Length).Fill(true);
        for (int s = 0; s < paths.Length; s++)
        {
            paths[s].AsSpan(0, fed * ps).CopyTo(values.AsSpan(context.Values.Length));
            var (loc, scale) = MoiraiPatches.Scale(values, observed);
            scales[s] = (loc, scale);
            int block = s * perPath;
            for (int r = 0; r < perPath; r++)
            {
                int row = block + r;
                for (int c = 0; c < ps; c++)
                {
                    patches[(row * ps) + c] = MoiraiPatches.Scaled(values[(r * ps) + c], loc, scale);
                }

                positions[row] = r < context.Rows ? context.Times[r] : lastTime + 1 + (r - context.Rows);
                // Causal in time (uni2ts' packed_causal_attention_mask): a row sees the rows of its path whose time id is at most its own.
                int end = r + 1;
                while (end < context.Rows && context.Times[end] <= positions[row])
                {
                    end++;
                }

                spans[2 * row] = block;
                spans[(2 * row) + 1] = block + end;
            }

            read[s] = block + perPath - 1;
        }

        var compiled = _compiled;
        compiled.SetInput<float>(Input(compiled, TimeSeriesGraphs.PatchesInput), patches);
        compiled.SetInput<int>(Input(compiled, TimeSeriesGraphs.PositionsInput), positions);
        compiled.SetInput<int>(Input(compiled, TimeSeriesGraphs.SpansInput), spans);
        compiled.SetInput<int>(Input(compiled, TimeSeriesGraphs.ReadRowsInput), read);
        var run = new RunParameters();
        run.Symbols[TimeSeriesGraphs.PatchSymbol] = rows;
        run.Symbols[TimeSeriesGraphs.ReadSymbol] = paths.Length;
        compiled.Run(run);
        int columns = _mixture.Columns;
        var output = new float[paths.Length * ps * columns];
        compiled.GetOutput<float>(TimeSeriesGraphs.DistributionOutput, output);
        var parameters = new double[paths.Length][][];
        for (int s = 0; s < paths.Length; s++)
        {
            parameters[s] = new double[ps][];
            for (int c = 0; c < ps; c++)
            {
                parameters[s][c] = _mixture.Map(output.AsSpan(((s * ps) + c) * columns, columns));
            }
        }

        return (scales, parameters);
    }

    private static Graph.Value Input(CompiledModel compiled, string name) => compiled.Graph.Inputs.First(v => v.Input!.Name == name);
}
