namespace Sharplight.Studio.Models;

/// <summary>
/// Navigation in a conversation's message tree: the active branch from the root to the current leaf, the versions
/// (siblings) of a message, and the newest leaf below a message.
/// </summary>
internal static class ConversationTree
{
    /// <summary>The messages of the branch ending at <see cref="Conversation.CurrentLeafId"/>, oldest first.</summary>
    public static IReadOnlyList<ChatMessageRecord> ActivePath(Conversation conversation)
    {
        return PathTo(conversation, conversation.CurrentLeafId);
    }

    /// <summary>The messages from the root to <paramref name="messageId"/> (inclusive), oldest first; empty for null.</summary>
    public static IReadOnlyList<ChatMessageRecord> PathTo(Conversation conversation, string? messageId)
    {
        var byId = conversation.Messages.ToDictionary(message => message.Id);
        var path = new List<ChatMessageRecord>();
        var visited = new HashSet<string>();
        var id = messageId;
        while (id is not null && byId.TryGetValue(id, out var message) && visited.Add(id))
        {
            path.Add(message);
            id = message.ParentId;
        }

        path.Reverse();
        return path;
    }

    /// <summary>The versions of a message: all messages with the same parent and role, oldest first.</summary>
    public static IReadOnlyList<ChatMessageRecord> Siblings(Conversation conversation, ChatMessageRecord message)
    {
        return [.. conversation.Messages
            .Where(other => other.ParentId == message.ParentId && other.Role == message.Role)
            .OrderBy(other => other.CreatedAt)];
    }

    /// <summary>The newest leaf reached from <paramref name="messageId"/> by always following the newest child.</summary>
    public static string LatestLeaf(Conversation conversation, string messageId)
    {
        var children = conversation.Messages.ToLookup(message => message.ParentId);
        var current = messageId;
        var visited = new HashSet<string> { current };
        while (true)
        {
            var newest = children[current].OrderByDescending(child => child.CreatedAt).FirstOrDefault();
            if (newest is null || !visited.Add(newest.Id))
            {
                return current;
            }

            current = newest.Id;
        }
    }

    /// <summary>Adds <paramref name="message"/> and makes it the current leaf.</summary>
    public static Conversation Append(Conversation conversation, ChatMessageRecord message)
    {
        return conversation with
        {
            Messages = [.. conversation.Messages, message],
            CurrentLeafId = message.Id,
            UpdatedAt = DateTimeOffset.UtcNow,
        };
    }

    /// <summary>Replaces the message with the same id.</summary>
    public static Conversation Replace(Conversation conversation, ChatMessageRecord message)
    {
        return conversation with
        {
            Messages = [.. conversation.Messages.Select(existing => existing.Id == message.Id ? message : existing)],
            UpdatedAt = DateTimeOffset.UtcNow,
        };
    }

    /// <summary>Shows the version <paramref name="offset"/> steps away from <paramref name="message"/> among its siblings.</summary>
    public static Conversation SwitchVersion(Conversation conversation, ChatMessageRecord message, int offset)
    {
        var siblings = Siblings(conversation, message);
        int index = siblings.ToList().FindIndex(sibling => sibling.Id == message.Id);
        int target = Math.Clamp(index + offset, 0, siblings.Count - 1);
        if (index < 0 || target == index)
        {
            return conversation;
        }

        return conversation with { CurrentLeafId = LatestLeaf(conversation, siblings[target].Id) };
    }

    /// <summary>A title from the first user message: its first line, shortened to <paramref name="maxLength"/> characters.</summary>
    public static string TitleFrom(string text, int maxLength = 48)
    {
        var line = text.Trim().Split('\n', 2)[0].Trim();
        if (line.Length == 0)
        {
            return "New chat";
        }

        return line.Length <= maxLength ? line : line[..(maxLength - 1)].TrimEnd() + "…";
    }
}
