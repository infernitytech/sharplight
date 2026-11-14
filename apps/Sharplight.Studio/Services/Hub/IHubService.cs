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

    /// <summary>Deletes the files of an interrupted pull (nothing when the model is stored).</summary>
    public void DiscardPartial(ModelReference reference);
}
