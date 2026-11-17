using Sharplight.Studio.Services.Updates;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>An update service that answers what a test says and records what was asked.</summary>
internal sealed class FakeUpdates : IUpdateService
{
    private readonly TaskCompletionSource _download = new();

    public UpdateSupport Support { get; set; } = UpdateSupport.Supported;

    public string CurrentVersion { get; set; } = "0.1.5";

    public string? Channel { get; set; } = "linux-x64-stable";

    public string? UpdateUrl { get; set; } = "https://updates.example.test/";

    /// <summary>What the next check returns.</summary>
    public UpdateCheckResult Result { get; set; } = UpdateCheckResult.UpToDate;

    /// <summary>When set, the download waits for <see cref="FinishDownload"/>.</summary>
    public bool HoldDownload { get; set; }

    public Exception? DownloadError { get; set; }

    public int Checks { get; private set; }

    public List<int> Reported { get; } = [];

    public UpdateOffer? Applied { get; private set; }

    /// <summary>What the check of a channel returns, by channel name; <see cref="Result"/> for the others.</summary>
    public Dictionary<string, UpdateCheckResult> ResultsByChannel { get; } = new(StringComparer.Ordinal);

    /// <summary>The channel of every check (null: the installed one).</summary>
    public List<string?> CheckedChannels { get; } = [];

    public Task<UpdateCheckResult> CheckAsync(string? channel, CancellationToken cancellationToken)
    {
        Checks++;
        CheckedChannels.Add(channel);
        return Task.FromResult(channel is not null && ResultsByChannel.TryGetValue(channel, out var result) ? result : Result);
    }

    public async Task DownloadAsync(UpdateOffer offer, IProgress<int> progress, CancellationToken cancellationToken)
    {
        progress.Report(40);
        Reported.Add(40);
        if (HoldDownload)
        {
            await _download.Task;
        }

        if (DownloadError is not null)
        {
            throw DownloadError;
        }
    }

    public void FinishDownload() => _download.TrySetResult();

    public void ApplyAndRestart(UpdateOffer offer) => Applied = offer;
}
