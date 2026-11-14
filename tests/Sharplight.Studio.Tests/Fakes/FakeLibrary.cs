using Sharplight.Studio.Services.Library;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A library with a fixed list.</summary>
internal sealed class FakeLibrary : IModelLibrary
{
    public List<LibraryEntry> Entries { get; } = [];

    public Task<IReadOnlyList<LibraryEntry>> ListAsync(CancellationToken cancellationToken) => Task.FromResult<IReadOnlyList<LibraryEntry>>([.. Entries]);

    public Task RemoveAsync(LibraryEntry entry)
    {
        Entries.Remove(entry);
        return Task.CompletedTask;
    }

    public Task AddLocalAsync(string path) => Task.CompletedTask;

    public Task<string> ConvertAsync(string source, string? projector, string destination, IProgress<string> log, CancellationToken cancellationToken) =>
        Task.FromResult(destination);
}
