using System.Text.Json;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the JSON configuration of a model component (<see cref="Models.ModelComponent.Config"/>) from scalars and arrays without
/// reflection-based serialization.
/// </summary>
internal sealed class ComponentConfigBuilder
{
    private readonly List<(string Key, Action<Utf8JsonWriter> Write)> _entries = [];

    /// <summary>Adds an integer.</summary>
    public ComponentConfigBuilder Add(string key, long value)
    {
        _entries.Add((key, w => w.WriteNumber(key, value)));
        return this;
    }

    /// <summary>Adds a number.</summary>
    public ComponentConfigBuilder Add(string key, double value)
    {
        _entries.Add((key, w => w.WriteNumber(key, value)));
        return this;
    }

    /// <summary>Adds a string.</summary>
    public ComponentConfigBuilder Add(string key, string value)
    {
        _entries.Add((key, w => w.WriteString(key, value)));
        return this;
    }

    /// <summary>Adds a boolean.</summary>
    public ComponentConfigBuilder Add(string key, bool value)
    {
        _entries.Add((key, w => w.WriteBoolean(key, value)));
        return this;
    }

    /// <summary>Adds an array of numbers.</summary>
    public ComponentConfigBuilder Add(string key, IReadOnlyList<float> values)
    {
        _entries.Add((key, w =>
        {
            w.WriteStartArray(key);
            foreach (float value in values)
            {
                w.WriteNumberValue(value);
            }

            w.WriteEndArray();
        }));
        return this;
    }

    /// <summary>The configuration as JSON values by key.</summary>
    public IReadOnlyDictionary<string, JsonElement> Build()
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            foreach (var (_, write) in _entries)
            {
                write(writer);
            }

            writer.WriteEndObject();
        }

        using var document = JsonDocument.Parse(buffer.ToArray());
        return document.RootElement.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.Clone(), StringComparer.Ordinal);
    }
}
