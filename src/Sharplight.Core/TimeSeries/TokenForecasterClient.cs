using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// The built-in <see cref="ITimeSeriesForecaster"/> over the token form of <see cref="TimeSeriesGraphs"/> (Chronos), following the chronos-forecasting pipeline:
/// the context is mean-scaled and quantized into bin tokens (plus the end token), encoded once, and every sample path is drawn token by token from the decoder
/// (temperature, top-k; the end token is suppressed); the tokens map back to the bin centers. Horizons beyond <see cref="PredictionLength"/> continue from the
/// context extended by the median path. Leading missing values are dropped (the reference masks them); missing values inside the context become padding tokens
/// that, unlike in the reference, stay visible to the attention.
/// </summary>
internal sealed class TokenForecasterClient : ITimeSeriesForecaster
{
    private const string OutputRowsSymbol = "R";
    private static readonly float[] s_levels = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
    private readonly TaskModel _model;
    private readonly CompiledModel _encoder;
    private readonly CompiledModel _decoder;
    private readonly Value _encoderTokens;
    private readonly Value _decoderTokens;
    private readonly Value _encoderOutput;
    private readonly Value? _sourceSpans;
    private readonly Value? _outputRows;
    private readonly float[] _logits;
    private readonly int _width;
    private readonly int _specialTokens;
    private readonly int _padToken;
    private readonly int _endToken;
    private readonly int _vocabulary;
    private readonly int _startToken;
    private readonly float[] _centers;
    private readonly float[] _boundaries;
    private readonly int _defaultSamples;
    private readonly float _defaultTemperature;
    private readonly int _defaultTopK;
    private readonly RunParameters _run = new();
    private readonly SemaphoreSlim _gate = new(1, 1);
    private float[] _source = [];      // the encoder output of the current context
    private int _sourceRows;
    private bool _sourceFresh;         // the next decoder pass writes the cross-attention states from _source
    private bool _disposed;

    public TokenForecasterClient(TaskModel model)
    {
        _model = model;
        var component = model.Description.Component(ModelComponentRoles.Encoder);
        ContextLength = component.ConfigValue(TimeSeriesGraphs.ContextLengthKey, 512);
        PredictionLength = component.ConfigValue(TimeSeriesGraphs.PredictionLengthKey, 64);
        _vocabulary = component.ConfigValue(TimeSeriesGraphs.BinCountKey, 4096);
        _specialTokens = component.ConfigValue(TimeSeriesGraphs.SpecialTokensKey, 2);
        _padToken = component.ConfigValue(TimeSeriesGraphs.PadTokenKey, 0);
        _endToken = component.ConfigValue(TimeSeriesGraphs.EndTokenKey, 1);
        _defaultSamples = component.ConfigValue(TimeSeriesGraphs.SampleCountKey, 20);
        _defaultTemperature = component.ConfigValue(TimeSeriesGraphs.TemperatureKey, 1f);
        _defaultTopK = component.ConfigValue(TimeSeriesGraphs.TopKKey, 50);
        (_centers, _boundaries) = Bins(component.ConfigValue(TimeSeriesGraphs.LowLimitKey, -15f), component.ConfigValue(TimeSeriesGraphs.HighLimitKey, 15f),
            _vocabulary - _specialTokens - 1);
        _startToken = Convert.ToInt32(model.Definition.Properties.GetValueOrDefault(WellKnown.DecoderStartToken, 0), System.Globalization.CultureInfo.InvariantCulture);
        _encoder = model.CompileComponent(ModelComponentRoles.Encoder);
        try
        {
            _decoder = model.CompileComponent(ModelComponentRoles.Decoder);
        }
        catch
        {
            _encoder.Dispose();
            throw;
        }

        _encoderTokens = _encoder.Graph.Input(InputRole.TokenIds);
        _decoderTokens = _decoder.Graph.Input(InputRole.TokenIds);
        _encoderOutput = _decoder.Graph.Input(InputRole.EncoderOutput);
        _sourceSpans = _decoder.Graph.TryInput(InputRole.RowSpans);
        _outputRows = _decoder.Graph.TryInput(InputRole.OutputRows);
        _width = (int)_encoder.Graph.Outputs[WellKnown.EncoderOutput].Shape[-1].Fixed;
        _logits = new float[(int)_decoder.Graph.Outputs[WellKnown.LogitsOutput].Shape[-1].Fixed];
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
        int horizon = options?.Horizon ?? PredictionLength;
        ArgumentOutOfRangeException.ThrowIfLessThan(horizon, 1, nameof(options));
        int samples = options?.SampleCount ?? _defaultSamples;
        ArgumentOutOfRangeException.ThrowIfLessThan(samples, 1, nameof(options));
        float temperature = options?.Temperature ?? _defaultTemperature;
        if (!(temperature > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "The temperature must be positive.");
        }

        int topK = options?.TopK ?? _defaultTopK;
        var levels = TimeSeriesQuantiles.Requested(options?.QuantileLevels, s_levels);
        var random = options?.Seed is { } seed ? new Random(seed) : new Random();
        var context = history.ToArray();
        return BackgroundCompute.RunAsync(_gate, () => Forecast(context, horizon, samples, new Sampling(temperature, topK, random), levels, cancellationToken), cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _decoder.Dispose();
        _encoder.Dispose();
        _gate.Dispose();
    }

    // torch.linspace(low, high, n) centers and the midpoints between them as bucket boundaries, with open outer buckets.
    private static (float[] Centers, float[] Boundaries) Bins(float low, float high, int count)
    {
        var centers = new float[count];
        double step = (high - (double)low) / (count - 1);
        for (int i = 0; i < count; i++)
        {
            centers[i] = i < count / 2 ? (float)(low + (i * step)) : (float)(high - ((count - 1 - i) * step));
        }

        var boundaries = new float[count + 1];
        boundaries[0] = -1e20f;
        boundaries[^1] = 1e20f;
        for (int i = 1; i < count; i++)
        {
            boundaries[i] = (centers[i - 1] + centers[i]) / 2;
        }

        return (centers, boundaries);
    }

    private TimeSeriesForecast Forecast(float[] history, int horizon, int sampleCount, Sampling sampling, IReadOnlyList<float> levels, CancellationToken cancellationToken)
    {
        var paths = new float[sampleCount][];
        for (int s = 0; s < sampleCount; s++)
        {
            paths[s] = new float[horizon];
        }

        var context = history;
        var median = new float[sampleCount];
        for (int done = 0; done < horizon;)
        {
            int steps = Math.Min(horizon - done, PredictionLength);
            float scale = Encode(context);
            for (int s = 0; s < sampleCount; s++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var tokens = Sample(steps, sampling, cancellationToken);
                for (int t = 0; t < steps; t++)
                {
                    int bin = Math.Clamp(tokens[t] - _specialTokens - 1, 0, _centers.Length - 1);
                    paths[s][done + t] = _centers[bin] * scale;
                }
            }

            done += steps;
            if (done < horizon)
            {
                // The next block continues from the context extended by the median path (torch.median: the lower middle sample).
                var extension = new float[steps];
                for (int t = 0; t < steps; t++)
                {
                    for (int s = 0; s < sampleCount; s++)
                    {
                        median[s] = paths[s][done - steps + t];
                    }

                    Array.Sort(median);
                    extension[t] = median[(sampleCount - 1) / 2];
                }

                context = [.. context, .. extension];
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

    // Tokenizes the last ContextLength values (leading missing values dropped), runs the encoder and hands its output to the first decoder pass; returns the scale.
    private float Encode(float[] history)
    {
        var context = history.Length > ContextLength ? history[^ContextLength..] : history;
        int first = 0;
        while (first < context.Length && float.IsNaN(context[first]))
        {
            first++;
        }

        context = context[first..];
        float scale = (float)TimeSeriesHistory.MeanAbsoluteScale(context);
        var tokens = new int[context.Length + (_endToken >= 0 ? 1 : 0)];
        for (int i = 0; i < context.Length; i++)
        {
            tokens[i] = float.IsNaN(context[i]) ? _padToken : Math.Clamp(Bucket(context[i] / scale) + _specialTokens, 0, _vocabulary - 1);
        }

        if (_endToken >= 0)
        {
            tokens[^1] = _endToken;
        }

        _encoder.SetInput<int>(_encoderTokens, tokens);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = tokens.Length;
        _encoder.Run(run);
        _source = new float[tokens.Length * _width];
        _encoder.GetOutput<float>(WellKnown.EncoderOutput, _source);
        _sourceRows = tokens.Length;
        _sourceFresh = true;
        return scale;
    }

    // torch.bucketize(right=True): the index of the first boundary above the value.
    private int Bucket(float value)
    {
        int low = 0;
        int high = _boundaries.Length;
        while (low < high)
        {
            int mid = (low + high) >>> 1;
            if (_boundaries[mid] <= value)
            {
                low = mid + 1;
            }
            else
            {
                high = mid;
            }
        }

        return low;
    }

    // One sample path of `steps` tokens from the decoder start token.
    private int[] Sample(int steps, Sampling sampling, CancellationToken cancellationToken)
    {
        var result = new int[steps];
        int token = _startToken;
        for (int position = 0; position < steps; position++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Step(token, position);
            if (_endToken >= 0)
            {
                _logits[_endToken] = float.NegativeInfinity;   // min_new_tokens = the steps: no early end
            }

            token = sampling.Next(_logits);
            result[position] = token;
        }

        return result;
    }

    private void Step(int token, int position)
    {
        _decoder.SetInput<int>(_decoderTokens, [token]);
        // The cross-attention keys and values are projected once per encoded context, on the first pass after Encode.
        int rows = _sourceFresh ? _sourceRows : 0;
        _decoder.SetInput<float>(_encoderOutput, _source.AsSpan(0, rows * _width));
        _sourceFresh = false;
        if (_sourceSpans is not null)
        {
            _decoder.SetInput<int>(_sourceSpans, [0, _sourceRows]);
        }

        if (_outputRows is not null)
        {
            _decoder.SetInput<int>(_outputRows, [0]);
        }

        _run.Symbols[WellKnown.TokenSymbol] = 1;
        _run.Symbols[WellKnown.EncoderSymbol] = rows;
        _run.Symbols[OutputRowsSymbol] = 1;
        _run.Scalars[WellKnown.PastScalar] = position;
        _run.Scalars[WellKnown.SourcePastScalar] = 0;
        _decoder.Run(_run);
        _decoder.GetOutput<float>(WellKnown.LogitsOutput, _logits);
    }
}
