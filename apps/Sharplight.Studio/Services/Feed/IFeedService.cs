using Sharplight.ModelFeed.Resolution;

namespace Sharplight.Studio.Services.Feed;

/// <summary>The Sharplight model feed: signed, preconverted <c>.slm</c> packages that install into the model store without conversion.</summary>
internal interface IFeedService
{
    /// <summary>The feed address in use (the setting, else the official feed).</summary>
    public string Url { get; }

    /// <summary>The channel in use (<c>stable</c> or <c>unstable</c>).</summary>
    public string Channel { get; }

    /// <summary>What package requirements are checked against on this machine.</summary>
    public CompatibilityEnvironment Environment { get; }

    /// <summary>
    /// Lists the channel. The index, key set and catalog pages are verified (signature, expiry, rollback, digests); when the feed cannot be
    /// reached the last verified copy is used. Never throws for feed problems: they are in <see cref="FeedListing.Failure"/>.
    /// </summary>
    /// <param name="cancellationToken">Cancels the request.</param>
    public Task<FeedListing> GetListingAsync(CancellationToken cancellationToken);

    /// <summary>
    /// Installs a package and its required dependencies into the model store: every manifest and package file is verified (signature, SHA-256 per
    /// chunk and as a whole), the download resumes after an interruption, and an installed version is replaced only after the new one verified.
    /// </summary>
    /// <param name="id">The package id.</param>
    /// <param name="progress">Receives progress on the calling thread of the download.</param>
    /// <param name="cancellationToken">Cancels the install; the partial download is kept for resuming.</param>
    /// <exception cref="Sharplight.ModelFeed.FeedVerificationException">A signature, digest or hash check failed: nothing was installed.</exception>
    /// <exception cref="FeedInstallException">No compatible version or a dependency is missing.</exception>
    /// <exception cref="IOException">The feed or the store could not be read or written.</exception>
    public Task<FeedInstallResult> InstallAsync(string id, IProgress<FeedInstallProgress>? progress, CancellationToken cancellationToken);
}
