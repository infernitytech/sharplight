using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Server;

/// <summary>The in-process HTTP server exposing the resident models over the OpenAI and Anthropic APIs.</summary>
internal interface IApiServer : IAsyncDisposable
{
    /// <summary>True while listening.</summary>
    public bool IsRunning { get; }

    /// <summary>The base address while running (e.g. <c>http://127.0.0.1:8080</c>).</summary>
    public string? BaseUrl { get; }

    /// <summary>Raised (on any thread) for every answered request.</summary>
    public event EventHandler<HttpLogEntry>? RequestLogged;

    /// <summary>Raised (on any thread) after the server started or stopped.</summary>
    public event EventHandler? StateChanged;

    /// <summary>Starts listening (restarts when already running). Requests route by their <c>model</c> field to any resident model the settings serve.</summary>
    /// <exception cref="InvalidOperationException">No API enabled, or the address is in use.</exception>
    public Task StartAsync(ServerSettings settings, CancellationToken cancellationToken);

    /// <summary>Stops listening; running requests get a few seconds to finish. The models stay loaded.</summary>
    public Task StopAsync();

    /// <summary>Restricts the API to these model names (empty = every model); applies to the running server at once.</summary>
    public void SetServedModels(IReadOnlyCollection<string> ids);
}
