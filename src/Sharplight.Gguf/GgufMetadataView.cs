namespace Sharplight.Gguf;

/// <summary>Read access to GGUF metadata of one architecture (<c>{gguf_arch}.key</c>).</summary>
internal sealed class GgufMetadataView(IReadOnlyDictionary<string, object> metadata, string ggufArchitecture, Func<string, GgufTensorInfo?>? tensors = null)
{
    /// <summary>All metadata of the file, with the full (architecture-prefixed) keys.</summary>
    public IReadOnlyDictionary<string, object> All => metadata;

    /// <summary>A tensor of the file by its GGUF name, or null when it is absent (or the view was made without tensor access).</summary>
    public GgufTensorInfo? Tensor(string name) => tensors?.Invoke(name);

    /// <summary>The <c>general.architecture</c> value; the prefix of every architecture-specific key.</summary>
    public string GgufArchitecture => ggufArchitecture;

    /// <summary>Whether <c>{arch}.{key}</c> exists.</summary>
    public bool Has(string key) => metadata.ContainsKey($"{ggufArchitecture}.{key}");

    /// <summary>The raw value of <c>{arch}.{key}</c>, or null when absent.</summary>
    public object? Raw(string key) => metadata.TryGetValue($"{ggufArchitecture}.{key}", out var v) ? v : null;

    /// <summary>A required integer value; throws <see cref="InvalidDataException"/> when absent.</summary>
    public int Int(string key) => Convert.ToInt32(Raw(key) ?? throw Missing(key));

    /// <summary>An integer value, or <paramref name="fallback"/> when absent.</summary>
    public int Int(string key, int fallback) => Raw(key) is { } v ? Convert.ToInt32(v) : fallback;

    /// <summary>A required floating-point value; throws <see cref="InvalidDataException"/> when absent.</summary>
    public float Float(string key) => Convert.ToSingle(Raw(key) ?? throw Missing(key));

    /// <summary>A floating-point value, or <paramref name="fallback"/> when absent.</summary>
    public float Float(string key, float fallback) => Raw(key) is { } v ? Convert.ToSingle(v) : fallback;

    /// <summary>A boolean value, or <paramref name="fallback"/> when absent.</summary>
    public bool Bool(string key, bool fallback) => Raw(key) is { } v ? Convert.ToBoolean(v) : fallback;

    private InvalidDataException Missing(string key) => new($"GGUF metadata {ggufArchitecture}.{key} missing.");
}
