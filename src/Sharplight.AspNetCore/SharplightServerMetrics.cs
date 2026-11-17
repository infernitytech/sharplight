using System.Diagnostics.Metrics;

namespace Sharplight.AspNetCore;

/// <summary>
/// <c>System.Diagnostics.Metrics</c> instruments of the API layer (meter <see cref="MeterName"/>), for OpenTelemetry or
/// <c>dotnet-counters</c>: requests, tokens and latency per API, model and scope. Tags: <c>sharplight.api</c> (the route),
/// <c>sharplight.model</c> (the alias), <c>sharplight.scope</c> (empty for the default scope), <c>http.response.status_code</c>
/// (requests and duration) and <c>sharplight.token.type</c> (<c>input</c>, <c>cached_input</c>, <c>output</c>).
/// </summary>
public sealed class SharplightServerMetrics : IDisposable
{
    /// <summary>The meter name.</summary>
    public const string MeterName = "Sharplight.AspNetCore";

    private readonly Meter _meter;
    private readonly bool _ownsMeter;
    private readonly Counter<long> _requests;
    private readonly Counter<long> _tokens;
    private readonly Histogram<double> _duration;
    private readonly Histogram<double> _timeToFirstToken;

    /// <summary>Creates the instruments on a meter of <paramref name="meterFactory"/> (or an own meter when null).</summary>
    /// <param name="meterFactory">The host's meter factory (registered by ASP.NET Core), or null.</param>
    public SharplightServerMetrics(IMeterFactory? meterFactory = null)
    {
        _ownsMeter = meterFactory is null;
        _meter = meterFactory?.Create(MeterName) ?? new Meter(MeterName);
        _requests = _meter.CreateCounter<long>("sharplight.server.requests", "{request}", "API requests completed.");
        _tokens = _meter.CreateCounter<long>("sharplight.server.tokens", "{token}", "Tokens processed, by type (input, cached_input, output).");
        _duration = _meter.CreateHistogram<double>("sharplight.server.request.duration", "s", "Time from the request reaching the endpoint to the end of the response.");
        _timeToFirstToken = _meter.CreateHistogram<double>("sharplight.server.time_to_first_token", "s", "Time to the first generated token of streamed responses.");
    }

    /// <summary>Records one completed request.</summary>
    /// <param name="usage">The request's usage.</param>
    public void Record(SharplightUsage usage)
    {
        ArgumentNullException.ThrowIfNull(usage);
        var api = new KeyValuePair<string, object?>("sharplight.api", usage.Api);
        var model = new KeyValuePair<string, object?>("sharplight.model", usage.Model ?? "");
        var scope = new KeyValuePair<string, object?>("sharplight.scope", usage.Scope ?? "");
        var status = new KeyValuePair<string, object?>("http.response.status_code", usage.StatusCode);
        _requests.Add(1, api, model, scope, status);
        _duration.Record(usage.Duration.TotalSeconds, api, model, scope, status);
        if (usage.TimeToFirstToken is { } ttft)
        {
            _timeToFirstToken.Record(ttft.TotalSeconds, api, model, scope);
        }

        AddTokens(usage.InputTokens, "input", api, model, scope);
        AddTokens(usage.CachedInputTokens, "cached_input", api, model, scope);
        AddTokens(usage.OutputTokens, "output", api, model, scope);
    }

    /// <summary>Disposes the meter when it is not the factory's.</summary>
    public void Dispose()
    {
        if (_ownsMeter)
        {
            _meter.Dispose();
        }
    }

    private void AddTokens(long count, string type, KeyValuePair<string, object?> api, KeyValuePair<string, object?> model, KeyValuePair<string, object?> scope)
    {
        if (count > 0)
        {
            _tokens.Add(count, api, model, scope, new KeyValuePair<string, object?>("sharplight.token.type", type));
        }
    }
}
