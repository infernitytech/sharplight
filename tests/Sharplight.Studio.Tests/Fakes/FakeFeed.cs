using Sharplight.ModelFeed.Resolution;
using Sharplight.Studio.Services.Feed;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A feed that lists nothing, for tests that do not use it.</summary>
internal sealed class FakeFeed : IFeedService
{
    public string Url => FeedOptions.DefaultUrl;

    public string Channel => FeedOptions.Stable;

    public CompatibilityEnvironment Environment { get; } = new();

    public Task<FeedListing> GetListingAsync(CancellationToken cancellationToken) => Task.FromResult(new FeedListing { Verified = true });

    public Task<FeedInstallResult> InstallAsync(string id, IProgress<FeedInstallProgress>? progress, CancellationToken cancellationToken)
        => throw new InvalidOperationException("The fake feed installs nothing.");
}
