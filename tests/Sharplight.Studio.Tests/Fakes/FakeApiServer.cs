using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Server;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A server that only tracks its state.</summary>
internal sealed class FakeApiServer : IApiServer
{
    public bool IsRunning { get; private set; }

    public string? BaseUrl { get; private set; }

    public event EventHandler<HttpLogEntry>? RequestLogged;

    public event EventHandler? StateChanged;

    public Task StartAsync(ServerSettings settings, CancellationToken cancellationToken)
    {
        IsRunning = true;
        BaseUrl = $"http://{settings.Host}:{settings.Port}";
        StateChanged?.Invoke(this, EventArgs.Empty);
        return Task.CompletedTask;
    }

    public Task StopAsync()
    {
        IsRunning = false;
        BaseUrl = null;
        StateChanged?.Invoke(this, EventArgs.Empty);
        return Task.CompletedTask;
    }

    public IReadOnlyCollection<string> Served { get; private set; } = [];

    public void SetServedModels(IReadOnlyCollection<string> ids) => Served = ids;

    public void Log(HttpLogEntry entry) => RequestLogged?.Invoke(this, entry);

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
