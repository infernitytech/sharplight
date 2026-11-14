using System.Diagnostics;
using System.Globalization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Sharplight.AspNetCore;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Server;

/// <summary>
/// <see cref="IApiServer"/> on Kestrel with the endpoints of <c>Sharplight.AspNetCore</c> (the same ones <c>sharplight-server</c>
/// maps) over the app's <see cref="ModelHost"/>: requests route by their <c>model</c> field to any resident model (loading sleeping
/// and, with on-demand loading, library models), chat and API requests share one engine per model, and stopping the server leaves
/// the models loaded.
/// </summary>
internal sealed class KestrelApiServer(ModelHost host) : IApiServer
{
    private static readonly TimeSpan s_stopTimeout = TimeSpan.FromSeconds(5);

    private WebApplication? _app;
    private volatile IReadOnlySet<string> _served = new HashSet<string>();

    /// <inheritdoc/>
    public bool IsRunning => _app is not null;

    /// <inheritdoc/>
    public string? BaseUrl { get; private set; }

    /// <inheritdoc/>
    public event EventHandler<HttpLogEntry>? RequestLogged;

    /// <inheritdoc/>
    public event EventHandler? StateChanged;

    /// <inheritdoc/>
    public void SetServedModels(IReadOnlyCollection<string> ids) => _served = ids.ToHashSet(StringComparer.Ordinal);

    /// <summary>The live filter: every model when the set is empty.</summary>
    internal bool IsServed(string name)
    {
        var served = _served;
        return served.Count == 0 || served.Contains(name);
    }

    /// <inheritdoc/>
    public async Task StartAsync(ServerSettings settings, CancellationToken cancellationToken)
    {
        await StopAsync().ConfigureAwait(false);
        if (!settings.OpenAi && !settings.Anthropic)
        {
            throw new InvalidOperationException("Enable the OpenAI or the Anthropic API.");
        }

        var url = $"http://{FormatHost(settings.Host)}:{settings.Port.ToString(CultureInfo.InvariantCulture)}";
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions
        {
            Args = ["--hostBuilder:reloadConfigOnChange=false"],
            ContentRootPath = AppContext.BaseDirectory,
        });
        builder.WebHost.UseUrls(url);
        builder.WebHost.ConfigureKestrel(kestrel => kestrel.Limits.MaxRequestBodySize = null);
        builder.Logging.ClearProviders();
        SetServedModels(settings.ServedModels);
        builder.Services.AddSharplight(host, options =>
        {
            options.OpenAi = settings.OpenAi;
            options.Anthropic = settings.Anthropic;
            if (!string.IsNullOrWhiteSpace(settings.ApiKey))
            {
                options.ApiKeys.Add(settings.ApiKey.Trim());
            }

            options.CorsOrigins.AddRange(ParseOrigins(settings.CorsOrigins));
            options.ModelFilter = IsServed;
            options.DrainModelsOnStop = false;
            options.Admin = false;
        });

        var app = builder.Build();
        app.Use(LogRequestAsync);
        app.UseSharplight();
        app.MapSharplight();
        try
        {
            await app.StartAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception error)
        {
            await app.DisposeAsync().ConfigureAwait(false);
            throw new InvalidOperationException($"Cannot listen on {url}: {error.GetBaseException().Message}", error);
        }

        _app = app;
        BaseUrl = url;
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    /// <inheritdoc/>
    public async Task StopAsync()
    {
        var app = _app;
        _app = null;
        BaseUrl = null;
        if (app is null)
        {
            return;
        }

        using var timeout = new CancellationTokenSource(s_stopTimeout);
        await app.StopAsync(timeout.Token).ConfigureAwait(false);
        await app.DisposeAsync().ConfigureAwait(false);
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>Stops the server.</summary>
    public async ValueTask DisposeAsync() => await StopAsync().ConfigureAwait(false);

    /// <summary>Comma or whitespace separated origins.</summary>
    internal static IReadOnlyList<string> ParseOrigins(string origins)
    {
        return [.. origins.Split([',', ' ', ';', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];
    }

    /// <summary>The host part of a URL (IPv6 addresses in brackets).</summary>
    internal static string FormatHost(string host)
    {
        var trimmed = string.IsNullOrWhiteSpace(host) ? "127.0.0.1" : host.Trim();
        return trimmed.Contains(':', StringComparison.Ordinal) && !trimmed.StartsWith('[') ? $"[{trimmed}]" : trimmed;
    }

    private async Task LogRequestAsync(HttpContext context, RequestDelegate next)
    {
        long start = Stopwatch.GetTimestamp();
        try
        {
            await next(context).ConfigureAwait(false);
        }
        finally
        {
            var entry = new HttpLogEntry(DateTimeOffset.UtcNow, context.Request.Method, context.Request.Path.Value ?? "/", context.Response.StatusCode,
                Stopwatch.GetElapsedTime(start).TotalMilliseconds, context.Connection.RemoteIpAddress?.ToString() ?? "");
            RequestLogged?.Invoke(this, entry);
        }
    }
}
