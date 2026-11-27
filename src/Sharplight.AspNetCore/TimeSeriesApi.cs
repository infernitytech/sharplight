using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.TimeSeries;

namespace Sharplight.AspNetCore;

/// <summary>
/// Probabilistic forecasting over HTTP (there is no OpenAI convention for it): <c>POST /v1/timeseries/forecast</c> with
/// <c>{ "model", "series": [[...], ...], "horizon", "quantile_levels", "samples", "temperature", "top_k", "seed", "patch_size" }</c> returns, per series, the
/// mean and the forecast of every quantile level. Missing values are <c>null</c>; a flat array of numbers is one series. The series are forecast
/// independently (batched into shared model passes where the model supports it, see <see cref="ITimeSeriesForecaster.ForecastBatchAsync"/>), unless the
/// request has <c>covariates</c> (<c>[{ "name", "history": [...], "future": [...] }]</c>) or sets <c>"joint": true</c>: then the series are the aligned targets of
/// one multivariate request (<see cref="TimeSeriesForecastRequest"/>, Chronos-2). Forecasters are registered with <see cref="AddSharplightTimeSeriesForecaster"/>,
/// or come from the server's <c>ModelHost</c> (task kind <c>time-series</c>) when <see cref="SharplightServerOptions.ServeHostModels"/> is on;
/// <c>MapSharplightServer</c> maps the endpoint when there is one.
/// </summary>
/// <example>
/// <code>
/// builder.Services.AddTimeSeriesForecaster(o => { o.ModelPath = "chronos-bolt-small"; o.Devices = [slot]; });
/// builder.Services.AddSharplightTimeSeriesForecaster(name: "chronos-bolt-small");   // the container's ITimeSeriesForecaster
/// var app = builder.Build();
/// app.MapSharplightTimeSeries();
/// // POST /v1/timeseries/forecast {"model": "chronos-bolt-small", "series": [[1, 2, null, 4, 5]], "horizon": 12, "quantile_levels": [0.1, 0.5, 0.9]}
/// // → {"object": "timeseries.forecast", "model": ..., "horizon": 12, "quantile_levels": [0.1, 0.5, 0.9],
/// //    "forecasts": [{"index": 0, "mean": [...], "quantiles": [[...], [...], [...]]}], "usage": {"series": 1, "observations": 4}}
/// </code>
/// </example>
public static class TimeSeriesApi
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Serves a forecaster on <c>/v1/timeseries/forecast</c> under <paramref name="name"/>: <paramref name="forecaster"/>, or the container's
        /// <see cref="ITimeSeriesForecaster"/> when it is <see langword="null"/> (resolved when the endpoint is mapped).
        /// </summary>
        /// <param name="forecaster">The forecaster, or <see langword="null"/> for the container's.</param>
        /// <param name="name">The model name clients pass.</param>
        /// <param name="configure">Adjusts the registered <see cref="TimeSeriesModels"/> (default model, limits), or <see langword="null"/>.</param>
        public IServiceCollection AddSharplightTimeSeriesForecaster(ITimeSeriesForecaster? forecaster = null, string name = "default",
            Action<TimeSeriesModels>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            var models = services.FirstOrDefault(d => d.ServiceType == typeof(TimeSeriesModels))?.ImplementationInstance as TimeSeriesModels;
            if (models is null)
            {
                models = new TimeSeriesModels();
                services.AddSingleton(models);
            }

            if (forecaster is not null)
            {
                models.Forecasters[name] = forecaster;
            }
            else
            {
                services.AddSingleton(new PendingTimeSeriesForecaster(name));
            }

            models.DefaultForecaster ??= name;
            configure?.Invoke(models);
            return services;
        }
    }

    extension(IEndpointRouteBuilder routes)
    {
        /// <summary>
        /// Maps <c>/v1/timeseries/forecast</c> for the registered forecasters, and for the host's when the server serves its models
        /// (<see cref="SharplightServerOptions.ServeHostModels"/>); nothing when there are none.
        /// </summary>
        public IEndpointRouteBuilder MapSharplightTimeSeries()
        {
            ArgumentNullException.ThrowIfNull(routes);
            bool host = HostTaskModels.Enabled(routes.ServiceProvider);
            var models = routes.ServiceProvider.GetService<TimeSeriesModels>();
            if (models is null && !host)
            {
                return routes;
            }

            models ??= new TimeSeriesModels();
            foreach (var pending in routes.ServiceProvider.GetServices<PendingTimeSeriesForecaster>())
            {
                models.Forecasters[pending.Name] = routes.ServiceProvider.GetRequiredService<ITimeSeriesForecaster>();
            }

            if (models.Forecasters.Count > 0 || host)
            {
                routes.MapPost("/v1/timeseries/forecast", http => Handle(http, () => Forecast(http, models)));
            }

            return routes;
        }
    }

    private static async Task Handle(HttpContext http, Func<Task> body)
    {
        try
        {
            await body();
        }
        catch (ApiException ex)
        {
            await ApiErrors.WriteAsync(http, ex);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
        {
            await ApiErrors.WriteAsync(http, 400, "invalid_request_error", ex.Message);
        }
    }

    private static async Task Forecast(HttpContext http, TimeSeriesModels models)
    {
        using var document = await HttpJson.ReadBodyAsync(http, models.MaxRequestBodyBytes);
        var body = document.RootElement;
        if (body.ValueKind != JsonValueKind.Object)
        {
            throw ApiException.BadRequest("The request body must be a JSON object.");
        }

        var series = ParseSeries(body, models.MaxSeries);
        var covariates = ParseCovariates(body);
        if (series.Count + covariates.Count > models.MaxSeries)
        {
            throw ApiException.BadRequest($"The request has {series.Count + covariates.Count} series; at most {models.MaxSeries} are accepted.", "series");
        }

        var options = ParseOptions(body, models);
        bool joint = (body.Bool("joint") ?? false) || covariates.Count > 0;
        using var leased = await Pick(http, models, body.Str("model"));
        var forecaster = leased.Client;
        IReadOnlyList<TimeSeriesForecast> forecasts;
        if (joint)
        {
            var request = new TimeSeriesForecastRequest { Targets = series, Covariates = covariates };
            forecasts = (await forecaster.ForecastAsync(request, options, http.RequestAborted)).Targets;
        }
        else
        {
            forecasts = await forecaster.ForecastBatchAsync(series, options, http.RequestAborted);
        }

        long observations = series.Sum(Observed) + covariates.Sum(c => Observed(c.History) + Observed(c.Future));
        SharplightRequestFeature.Of(http)?.AddInput(observations);
        var data = new JsonArray();
        for (int i = 0; i < forecasts.Count; i++)
        {
            var forecast = forecasts[i];
            data.Add(new JsonObject
            {
                ["index"] = i,
                ["mean"] = Values(forecast.Mean),
                ["quantiles"] = new JsonArray([.. forecast.Quantiles.Select(level => (JsonNode)Values(level))]),
            });
        }

        var first = forecasts[0];
        await HttpJson.WriteAsync(http, new JsonObject
        {
            ["id"] = "fcst-" + Guid.NewGuid().ToString("N"),
            ["object"] = "timeseries.forecast",
            ["created"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            ["model"] = leased.Name,
            ["horizon"] = first.Horizon,
            ["quantile_levels"] = new JsonArray([.. first.QuantileLevels.Select(level => (JsonNode)JsonValue.Create(level))]),
            ["forecasts"] = data,
            ["usage"] = new JsonObject { ["series"] = series.Count + covariates.Count, ["observations"] = observations },
        });
    }

    private static long Observed(ReadOnlyMemory<float> values)
    {
        long count = 0;
        foreach (float value in values.Span)
        {
            if (!float.IsNaN(value))
            {
                count++;
            }
        }

        return count;
    }

    // A forecast array; non-finite values (a diverging model) become null, which JSON can carry.
    private static JsonArray Values(float[] values)
        => new([.. values.Select(value => float.IsFinite(value) ? JsonValue.Create(value) : null)]);

    /// <summary><c>series</c>: an array of series (arrays of numbers or null), or one series as a flat array.</summary>
    private static List<ReadOnlyMemory<float>> ParseSeries(JsonElement body, int maxSeries)
    {
        const string Shape = "'series' must be an array of series, each an array of numbers (null for a missing value), or a single such array.";
        if (body.Get("series") is not { ValueKind: JsonValueKind.Array } series)
        {
            throw ApiException.BadRequest("'series' is required. " + Shape, "series");
        }

        if (series.GetArrayLength() == 0)
        {
            throw ApiException.BadRequest("'series' must not be empty.", "series");
        }

        if (series[0].ValueKind is JsonValueKind.Number or JsonValueKind.Null)
        {
            return [Numbers(series, "series", Shape)];
        }

        if (series.GetArrayLength() > maxSeries)
        {
            throw ApiException.BadRequest($"'series' has {series.GetArrayLength()} series; at most {maxSeries} are accepted.", "series");
        }

        var result = new List<ReadOnlyMemory<float>>(series.GetArrayLength());
        foreach (var item in series.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Array)
            {
                throw ApiException.BadRequest(Shape, "series");
            }

            result.Add(Numbers(item, "series", Shape));
        }

        return result;
    }

    private static List<TimeSeriesCovariate> ParseCovariates(JsonElement body)
    {
        const string Shape = "Each covariate must be an object with a 'history' array (numbers or null) and optionally 'name' and 'future'.";
        var result = new List<TimeSeriesCovariate>();
        foreach (var item in body.Arr("covariates"))
        {
            if (item.ValueKind != JsonValueKind.Object || item.Get("history") is not { ValueKind: JsonValueKind.Array } history)
            {
                throw ApiException.BadRequest(Shape, "covariates");
            }

            var future = item.Get("future") switch
            {
                null => ReadOnlyMemory<float>.Empty,
                { ValueKind: JsonValueKind.Array } values => Numbers(values, "covariates", Shape),
                _ => throw ApiException.BadRequest(Shape, "covariates"),
            };
            result.Add(new TimeSeriesCovariate
            {
                Name = item.Str("name") ?? $"covariate_{result.Count}",
                History = Numbers(history, "covariates", Shape),
                Future = future,
            });
        }

        return result;
    }

    private static float[] Numbers(JsonElement array, string param, string shape)
    {
        var values = new float[array.GetArrayLength()];
        int i = 0;
        foreach (var item in array.EnumerateArray())
        {
            values[i++] = item.ValueKind switch
            {
                JsonValueKind.Null => float.NaN,
                JsonValueKind.Number when item.TryGetDouble(out double value) && Math.Abs(value) <= float.MaxValue => (float)value,
                JsonValueKind.Number => throw ApiException.BadRequest($"'{param}' holds a value outside the float range.", param),
                _ => throw ApiException.BadRequest(shape, param),
            };
        }

        return values;
    }

    private static TimeSeriesForecastOptions ParseOptions(JsonElement body, TimeSeriesModels models)
    {
        int? horizon = body.Int("horizon");
        if (horizon is int h && (h < 1 || h > models.MaxHorizon))
        {
            throw ApiException.BadRequest($"'horizon' must be between 1 and {models.MaxHorizon}.", "horizon");
        }

        int? samples = body.Int("samples");
        if (samples is int n && (n < 1 || n > models.MaxSamples))
        {
            throw ApiException.BadRequest($"'samples' must be between 1 and {models.MaxSamples}.", "samples");
        }

        float[]? levels = null;
        if (body.Get("quantile_levels") is { } list)
        {
            if (list.ValueKind != JsonValueKind.Array || list.GetArrayLength() == 0)
            {
                throw ApiException.BadRequest("'quantile_levels' must be a non-empty array of numbers between 0 and 1.", "quantile_levels");
            }

            levels = [.. list.EnumerateArray().Select(level => level.ValueKind == JsonValueKind.Number && level.GetDouble() is > 0 and < 1
                ? level.GetSingle()
                : throw ApiException.BadRequest("Every quantile level must be a number above 0 and below 1.", "quantile_levels"))];
        }

        float? temperature = body.Float("temperature");
        if (temperature is float t && !(t > 0))
        {
            throw ApiException.BadRequest("'temperature' must be above 0.", "temperature");
        }

        int? topK = body.Int("top_k");
        if (topK < 0)
        {
            throw ApiException.BadRequest("'top_k' must be 0 (none) or more.", "top_k");
        }

        long? seed = body.Long("seed");
        if (seed is long s && (s < int.MinValue || s > int.MaxValue))
        {
            throw ApiException.BadRequest("'seed' must be a 32-bit integer.", "seed");
        }

        return new TimeSeriesForecastOptions
        {
            Horizon = horizon,
            QuantileLevels = levels,
            SampleCount = samples,
            Temperature = temperature,
            TopK = topK,
            PatchSize = PatchSize(body),
            Seed = (int?)seed,
        };
    }

    // "patch_size": an integer, or "auto" (TimeSeriesForecastOptions.AutoPatchSize: the patch size of the lowest validation loss).
    private static int? PatchSize(JsonElement body)
        => body.Get("patch_size") is { ValueKind: JsonValueKind.String } text
            ? string.Equals(text.GetString(), "auto", StringComparison.OrdinalIgnoreCase)
                ? TimeSeriesForecastOptions.AutoPatchSize
                : throw ApiException.BadRequest("'patch_size' must be an integer or \"auto\".", "patch_size")
            : body.Int("patch_size");

    private static async Task<Leased> Pick(HttpContext http, TimeSeriesModels models, string? requested)
    {
        if (requested is not null && models.Forecasters.TryGetValue(requested, out var named))
        {
            await SharplightServerContext.AdmitAsync(http, requested);
            return new Leased(named, null, requested);
        }

        if (await HostTaskModels.TryAcquireAsync<ITimeSeriesForecaster>(http, requested, "time-series forecasting") is { } hosted)
        {
            return new Leased(hosted.Client, hosted.Lease, requested!);
        }

        // A single registered model answers to any name.
        if (models.Forecasters.Count > 0 && (requested is null || models.Forecasters.Count == 1))
        {
            string name = models.DefaultForecaster is { } fallback && models.Forecasters.ContainsKey(fallback) ? fallback : models.Forecasters.Keys.First();
            await SharplightServerContext.AdmitAsync(http, requested ?? name);
            return new Leased(models.Forecasters[name], null, requested ?? name);
        }

        string available = models.Forecasters.Count > 0 ? $" (available: {string.Join(", ", models.Forecasters.Keys)})" : "";
        throw new ApiException(404, "invalid_request_error", requested is null
            ? "'model' is required: name the forecasting model." : $"The time-series model '{requested}' does not exist{available}.", "model_not_found", "model");
    }

    private readonly record struct Leased(ITimeSeriesForecaster Client, IDisposable? Lease, string Name) : IDisposable
    {
        public void Dispose() => Lease?.Dispose();
    }
}
