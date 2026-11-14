using Sharplight.Hub;

namespace Sharplight.Studio.Services.Hub;

/// <summary>Hugging Face search, repository details and pulls into the model store.</summary>
internal interface IHubService
{
    /// <summary>Searches model repositories, most downloaded first.</summary>
    public Task<IReadOnlyList<HubModelSummary>> SearchAsync(HubSearchQuery query, CancellationToken cancellationToken);

    /// <summary>Files and variants of a repository.</summary>
    public Task<RepositoryDetails> GetDetailsAsync(string repository, CancellationToken cancellationToken);

    /// <summary>
    /// Downloads, converts (and optionally precompiles) a model into the store. Cancelling keeps the downloaded parts, so
    /// pulling again resumes; <see cref="DiscardPartial"/> deletes them.
    /// </summary>
    /// <returns>The stored model.</returns>
    public Task<ModelManifest> PullAsync(ModelReference reference, IProgress<PullUpdate> progress, CancellationToken cancellationToken);

    /// <summary>Deletes the files of an interrupted pull (nothing when the model is stored).</summary>
    public void DiscardPartial(ModelReference reference);
}
