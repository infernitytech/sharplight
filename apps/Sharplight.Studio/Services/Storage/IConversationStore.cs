using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Storage;

/// <summary>Persists conversations.</summary>
internal interface IConversationStore
{
    /// <summary>Every conversation, most recently changed first.</summary>
    public Task<IReadOnlyList<ConversationSummary>> ListAsync();

    /// <summary>One conversation with its messages and attachments, or null.</summary>
    public Task<Conversation?> LoadAsync(string id);

    /// <summary>Inserts or replaces a conversation.</summary>
    public Task SaveAsync(Conversation conversation);

    /// <summary>Deletes a conversation and its attachments.</summary>
    public Task DeleteAsync(string id);
}
