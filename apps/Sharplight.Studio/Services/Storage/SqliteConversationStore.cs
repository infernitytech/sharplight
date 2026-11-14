using System.Text.Json;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Storage;

/// <summary>
/// <see cref="IConversationStore"/> in SQLite: each conversation is one JSON document (messages and attachment metadata),
/// image bytes live in their own table so listing and saving stay cheap.
/// </summary>
internal sealed class SqliteConversationStore : IConversationStore
{
    private readonly StudioDatabase _database;

    public SqliteConversationStore(StudioDatabase database)
    {
        _database = database;
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<ConversationSummary>> ListAsync()
    {
        return Task.Run<IReadOnlyList<ConversationSummary>>(() =>
        {
            using var connection = _database.Open();
            using var command = connection.CreateCommand();
            command.CommandText = "SELECT id, title, updated, message_count FROM conversations ORDER BY updated DESC";
            using var reader = command.ExecuteReader();
            var list = new List<ConversationSummary>();
            while (reader.Read())
            {
                list.Add(new ConversationSummary(reader.GetString(0), reader.GetString(1), DateTimeOffset.FromUnixTimeMilliseconds(reader.GetInt64(2)), reader.GetInt32(3)));
            }

            return list;
        });
    }

    /// <inheritdoc/>
    public Task<Conversation?> LoadAsync(string id)
    {
        return Task.Run(() =>
        {
            using var connection = _database.Open();
            using var command = connection.CreateCommand();
            command.CommandText = "SELECT body FROM conversations WHERE id = $id";
            command.Parameters.AddWithValue("$id", id);
            if (command.ExecuteScalar() is not string body)
            {
                return null;
            }

            var conversation = JsonSerializer.Deserialize(body, StudioJsonContext.Default.Conversation);
            if (conversation is null)
            {
                return null;
            }

            var data = new Dictionary<string, byte[]>();
            using var attachments = connection.CreateCommand();
            attachments.CommandText = "SELECT id, data FROM attachment_data WHERE conversation_id = $id";
            attachments.Parameters.AddWithValue("$id", id);
            using var reader = attachments.ExecuteReader();
            while (reader.Read())
            {
                data[reader.GetString(0)] = (byte[])reader.GetValue(1);
            }

            return conversation with
            {
                Messages = [.. conversation.Messages.Select(message => message.Attachments.Count == 0 ? message : message with
                {
                    Attachments = [.. message.Attachments.Select(attachment => data.TryGetValue(attachment.Id, out var bytes) ? attachment with { Data = bytes } : attachment)],
                })],
            };
        });
    }

    /// <inheritdoc/>
    public Task SaveAsync(Conversation conversation)
    {
        return Task.Run(() =>
        {
            var stripped = conversation with
            {
                Messages = [.. conversation.Messages.Select(message => message.Attachments.Count == 0 ? message : message with
                {
                    Attachments = [.. message.Attachments.Select(attachment => attachment with { Data = null })],
                })],
            };
            using var connection = _database.Open();
            using var transaction = connection.BeginTransaction();
            using (var command = connection.CreateCommand())
            {
                command.CommandText = """
                    INSERT INTO conversations (id, title, updated, message_count, body) VALUES ($id, $title, $updated, $count, $body)
                    ON CONFLICT(id) DO UPDATE SET title = $title, updated = $updated, message_count = $count, body = $body
                    """;
                command.Parameters.AddWithValue("$id", conversation.Id);
                command.Parameters.AddWithValue("$title", conversation.Title);
                command.Parameters.AddWithValue("$updated", conversation.UpdatedAt.ToUnixTimeMilliseconds());
                command.Parameters.AddWithValue("$count", conversation.Messages.Count);
                command.Parameters.AddWithValue("$body", JsonSerializer.Serialize(stripped, StudioJsonContext.Default.Conversation));
                command.ExecuteNonQuery();
            }

            foreach (var attachment in conversation.Messages.SelectMany(message => message.Attachments).Where(attachment => attachment.Data is not null))
            {
                using var command = connection.CreateCommand();
                command.CommandText = "INSERT OR IGNORE INTO attachment_data (id, conversation_id, data) VALUES ($id, $conversation, $data)";
                command.Parameters.AddWithValue("$id", attachment.Id);
                command.Parameters.AddWithValue("$conversation", conversation.Id);
                command.Parameters.AddWithValue("$data", attachment.Data!);
                command.ExecuteNonQuery();
            }

            transaction.Commit();
        });
    }

    /// <inheritdoc/>
    public Task DeleteAsync(string id)
    {
        return Task.Run(() =>
        {
            using var connection = _database.Open();
            using var command = connection.CreateCommand();
            command.CommandText = "DELETE FROM conversations WHERE id = $id; DELETE FROM attachment_data WHERE conversation_id = $id";
            command.Parameters.AddWithValue("$id", id);
            command.ExecuteNonQuery();
        });
    }
}
