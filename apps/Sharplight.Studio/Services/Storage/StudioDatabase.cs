using Microsoft.Data.Sqlite;

namespace Sharplight.Studio.Services.Storage;

/// <summary>The SQLite database holding conversations and the statistics history; creates the schema on first use.</summary>
internal sealed class StudioDatabase
{
    private const string Schema = """
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            updated INTEGER NOT NULL,
            message_count INTEGER NOT NULL,
            body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attachment_data (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            data BLOB NOT NULL);
        CREATE INDEX IF NOT EXISTS attachment_conversation ON attachment_data (conversation_id);
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            model TEXT NOT NULL,
            source INTEGER NOT NULL,
            device TEXT NOT NULL,
            prompt INTEGER NOT NULL,
            cached INTEGER NOT NULL,
            completion INTEGER NOT NULL,
            queue_ms REAL NOT NULL,
            prefill_ms REAL NOT NULL,
            decode_ms REAL NOT NULL,
            total_ms REAL NOT NULL,
            draft INTEGER NOT NULL,
            accepted INTEGER NOT NULL,
            finish TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts);
        """;

    private readonly string _connectionString;

    /// <summary>Opens (creating) the database file.</summary>
    public StudioDatabase(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        _connectionString = new SqliteConnectionStringBuilder { DataSource = path, Pooling = true }.ToString();
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = Schema;
        command.ExecuteNonQuery();
    }

    /// <summary>An open connection (pooled; dispose it after use).</summary>
    public SqliteConnection Open()
    {
        var connection = new SqliteConnection(_connectionString);
        connection.Open();
        return connection;
    }
}
