namespace Sharplight.Studio.Services.Library;

/// <summary>The models on disk: the store plus files the user added.</summary>
internal interface IModelLibrary
{
    /// <summary>Every model, store first.</summary>
    public Task<IReadOnlyList<LibraryEntry>> ListAsync(CancellationToken cancellationToken);

    /// <summary>Deletes a stored model, or removes an added file from the list (the file itself is kept).</summary>
    public Task RemoveAsync(LibraryEntry entry);

    /// <summary>Adds a model file or checkpoint directory outside the store.</summary>
    public Task AddLocalAsync(string path);

    /// <summary>Converts a model file (with an optional vision projector) to a <c>.slm</c> package and adds the package.</summary>
    /// <returns>The package path.</returns>
    public Task<string> ConvertAsync(string source, string? projector, string destination, IProgress<string> log, CancellationToken cancellationToken);
}
