using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Monitoring;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>Platform, dispatcher, theme and monitor stand-ins (UI work runs inline).</summary>
internal sealed class FakePlatform : IPlatformServices, IUiDispatcher, IThemeService, ISystemMonitor
{
    public List<string> Copied { get; } = [];

    public List<string> Opened { get; } = [];

    public Func<SystemSample>? Sampler { get; set; }

    public Task<IReadOnlyList<string>> PickFilesAsync(string title, IReadOnlyList<FileFilter> filters, bool allowMultiple) =>
        Task.FromResult<IReadOnlyList<string>>([]);

    public Task<string?> PickFolderAsync(string title) => Task.FromResult<string?>(null);

    public Task<string?> PickSaveFileAsync(string title, string suggestedName, FileFilter filter) => Task.FromResult<string?>(null);

    public Task CopyTextAsync(string text)
    {
        Copied.Add(text);
        return Task.CompletedTask;
    }

    public void Open(string pathOrUrl) => Opened.Add(pathOrUrl);

    public void Post(Action action) => action();

    public void Apply(AppTheme theme, string accent)
    {
    }

    public SystemSample Sample() => Sampler?.Invoke() ?? new SystemSample(DateTimeOffset.UtcNow, 12, 1L << 30, 8L << 30, [], 0, 0, 0, 0);
}
