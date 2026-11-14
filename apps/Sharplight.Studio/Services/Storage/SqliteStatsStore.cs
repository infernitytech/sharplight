using Microsoft.Data.Sqlite;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Storage;

/// <summary><see cref="IStatsStore"/> in SQLite, one row per request.</summary>
internal sealed class SqliteStatsStore : IStatsStore
{
    private readonly StudioDatabase _database;

    public SqliteStatsStore(StudioDatabase database)
    {
        _database = database;
    }

    /// <inheritdoc/>
    public event EventHandler<RequestRecord>? RecordAdded;

    /// <inheritdoc/>
    public async Task AddAsync(RequestRecord record)
    {
        long id = await Task.Run(() =>
        {
            using var connection = _database.Open();
            using var command = connection.CreateCommand();
            command.CommandText = """
                INSERT INTO requests (ts, model, source, device, prompt, cached, completion, queue_ms, prefill_ms, decode_ms, total_ms, draft, accepted, finish)
                VALUES ($ts, $model, $source, $device, $prompt, $cached, $completion, $queue, $prefill, $decode, $total, $draft, $accepted, $finish);
                SELECT last_insert_rowid();
                """;
            var s = record.Stats;
            command.Parameters.AddWithValue("$ts", record.Timestamp.ToUnixTimeMilliseconds());
            command.Parameters.AddWithValue("$model", record.Model);
            command.Parameters.AddWithValue("$source", (int)record.Source);
            command.Parameters.AddWithValue("$device", record.Device);
            command.Parameters.AddWithValue("$prompt", s.PromptTokens);
            command.Parameters.AddWithValue("$cached", s.CachedTokens);
            command.Parameters.AddWithValue("$completion", s.CompletionTokens);
            command.Parameters.AddWithValue("$queue", s.QueueMs);
            command.Parameters.AddWithValue("$prefill", s.PrefillMs);
            command.Parameters.AddWithValue("$decode", s.DecodeMs);
            command.Parameters.AddWithValue("$total", s.TotalMs);
            command.Parameters.AddWithValue("$draft", s.DraftTokens);
            command.Parameters.AddWithValue("$accepted", s.AcceptedDraftTokens);
            command.Parameters.AddWithValue("$finish", s.Finish);
            return (long)command.ExecuteScalar()!;
        }).ConfigureAwait(false);
        RecordAdded?.Invoke(this, record with { Id = id });
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<RequestRecord>> QueryAsync(DateTimeOffset since, string? model)
    {
        return Task.Run<IReadOnlyList<RequestRecord>>(() =>
        {
            using var connection = _database.Open();
            using var command = connection.CreateCommand();
            command.CommandText = "SELECT id, ts, model, source, device, prompt, cached, completion, queue_ms, prefill_ms, decode_ms, total_ms, draft, accepted, finish "
                + "FROM requests WHERE ts >= $since" + (model is null ? "" : " AND model = $model") + " ORDER BY ts";
            command.Parameters.AddWithValue("$since", since.ToUnixTimeMilliseconds());
            if (model is not null)
            {
                command.Parameters.AddWithValue("$model", model);
            }

            using var reader = command.ExecuteReader();
            var list = new List<RequestRecord>();
            while (reader.Read())
            {
                list.Add(Read(reader));
            }

            return list;
        });
    }

    /// <inheritdoc/>
    public Task ClearAsync()
    {
        return Task.Run(() =>
        {
            using var connection = _database.Open();
            using var command = connection.CreateCommand();
            command.CommandText = "DELETE FROM requests";
            command.ExecuteNonQuery();
        });
    }

    private static RequestRecord Read(SqliteDataReader reader)
    {
        return new RequestRecord
        {
            Id = reader.GetInt64(0),
            Timestamp = DateTimeOffset.FromUnixTimeMilliseconds(reader.GetInt64(1)),
            Model = reader.GetString(2),
            Source = (RequestSource)reader.GetInt32(3),
            Device = reader.GetString(4),
            Stats = new MessageStats
            {
                PromptTokens = reader.GetInt32(5),
                CachedTokens = reader.GetInt32(6),
                CompletionTokens = reader.GetInt32(7),
                QueueMs = reader.GetDouble(8),
                PrefillMs = reader.GetDouble(9),
                DecodeMs = reader.GetDouble(10),
                TotalMs = reader.GetDouble(11),
                DraftTokens = reader.GetInt32(12),
                AcceptedDraftTokens = reader.GetInt32(13),
                Finish = reader.GetString(14),
            },
        };
    }
}
