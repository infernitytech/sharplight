using Sharplight.Hub;
using Sharplight.Hub.Catalog;

namespace Sharplight.Studio.Services.Hub;

/// <summary>Hugging Face search, repository details and pulls into the model store.</summary>
internal interface IHubService
{
    /// <summary>Searches model repositories, most downloaded first.</summary>
    public Task<IReadOnlyList<HubModelSummary>> SearchAsync(HubSearchQuery query, CancellationToken cancellationToken);

    /// <summary>
    /// The models Discover shows before anything is searched (trending, new, most downloaded), cached for a few hours (see
    /// <see cref="HighlightsCatalog"/>): without a task the language models whose architecture Sharplight supports, with a task that
    /// task's models.
    /// </summary>
    /// <param name="task">The model type to highlight, or null for the default highlights.</param>
    /// <param name="cancellationToken">Cancels the requests.</param>
    public Task<IReadOnlyList<HighlightSection>> GetHighlightsAsync(ModelTask? task, CancellationToken cancellationToken);

    /// <summary>Files and variants of a repository.</summary>
    public Task<RepositoryDetails> GetDetailsAsync(string repository, CancellationToken cancellationToken);

    /// <summary>
    /// Reads a variant's metadata without downloading it (range requests for the start of a GGUF file and, with
    /// <paramref name="readWeightFormats"/>, its tensor table; a checkpoint's <c>config.json</c>; an ONNX model's opset imports): its
    /// dimensions for the fit estimate and its likely compatibility. Never throws for unreadable metadata; the verdict is then unknown.
    /// </summary>
    /// <param name="details">The repository.</param>
    /// <param name="variant">The variant.</param>
    /// <param name="readWeightFormats">Also read the GGUF tensor table (up to a few tens of megabytes from the start of the file).</param>
    /// <param name="cancellationToken">Cancels the requests.</param>
    public Task<ModelProbe> ProbeAsync(RepositoryDetails details, VariantInfo variant, bool readWeightFormats, CancellationToken cancellationToken);

    /// <summary>
    /// Downloads, converts (and optionally precompiles) a model into the store. Cancelling keeps the downloaded parts, so
    /// pulling again resumes; <see cref="DiscardPartial"/> deletes them.
    /// </summary>
    /// <returns>The stored model.</returns>
    public Task<ModelManifest> PullAsync(ModelReference reference, IProgress<PullUpdate> progress, CancellationToken cancellationToken);

    /// <summary>
    /// The verified-model catalog (<see cref="VerifiedCatalogSource"/>): the embedded copy or a newer published one; never throws for
    /// network failures.
    /// </summary>
    public Task<VerifiedCatalog> GetCatalogAsync(CancellationToken cancellationToken);

    /// <summary>
    /// Installs a catalog variant with its companions (<see cref="ModelPuller.InstallAsync"/>): the pinned files, one package. Cancelling keeps
    /// the downloaded parts, like <see cref="PullAsync"/>; <see cref="DiscardPartial"/> with <see cref="CatalogInstallPlan.Reference"/> deletes them.
    /// </summary>
    /// <returns>The stored model.</returns>
    public Task<ModelManifest> InstallAsync(CatalogInstallPlan plan, IProgress<PullUpdate> progress, CancellationToken cancellationToken);

    /// <summary>Deletes the files of an interrupted pull (nothing when the model is stored).</summary>
    public void DiscardPartial(ModelReference reference);
}
