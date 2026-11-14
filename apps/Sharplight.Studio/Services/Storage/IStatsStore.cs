using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Storage;

/// <summary>The history of finished generation requests.</summary>
internal interface IStatsStore
{
    /// <summary>Raised (on any thread) after a record was added.</summary>
    public event EventHandler<RequestRecord>? RecordAdded;

    /// <summary>Stores a record.</summary>
    public Task AddAsync(RequestRecord record);

    /// <summary>Records since <paramref name="since"/>, optionally of one model, oldest first.</summary>
    public Task<IReadOnlyList<RequestRecord>> QueryAsync(DateTimeOffset since, string? model);

    /// <summary>Deletes the whole history.</summary>
    public Task ClearAsync();
}
