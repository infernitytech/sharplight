using Sharplight.Studio.Services.Library;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A library with a fixed list.</summary>
internal sealed class FakeLibrary : IModelLibrary
{
    public List<LibraryEntry> Entries { get; } = [];

    public Task<IReadOnlyList<LibraryEntry>> ListAsync(CancellationToken cancellationToken) => Task.FromResult<IReadOnlyList<LibraryEntry>>([.. Entries]);

    /// <summary>Entry returned by <see cref="ReloadAsync"/>, or null to return the entry unchanged.</summary>
    public Func<LibraryEntry, LibraryEntry>? Reload { get; set; }

    public Task<LibraryEntry> ReloadAsync(LibraryEntry entry, CancellationToken cancellationToken) => Task.FromResult(Reload?.Invoke(entry) ?? entry);

    public Task RemoveAsync(LibraryEntry entry)
    {
        Entries.Remove(entry);
        return Task.CompletedTask;
    }

    public Task AddLocalAsync(string path) => Task.CompletedTask;

    public Task<string> ConvertAsync(string source, string? projector, string destination, IProgress<string> log, CancellationToken cancellationToken) =>
        Task.FromResult(destination);
}
