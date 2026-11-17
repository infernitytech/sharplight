using System.Text.Json.Serialization;

namespace Sharplight.Hub.Catalog;

/// <summary>How a catalog entry was checked (the levels of docs/compatible-models.md).</summary>
[JsonConverter(typeof(JsonStringEnumConverter<CatalogLevel>))]
public enum CatalogLevel
{
    /// <summary>Output compared with a reference implementation on the real weights.</summary>
    [JsonStringEnumMemberName("verified")]
    Verified,

    /// <summary>Works with a stated limitation (see the entry's notes).</summary>
    [JsonStringEnumMemberName("partial")]
    Partial,

    /// <summary>The real checkpoint imports; generation was not compared with a reference.</summary>
    [JsonStringEnumMemberName("loads")]
    Loads,

    /// <summary>Only the architecture was verified, on a small random checkpoint of the same structure.</summary>
    [JsonStringEnumMemberName("verified-tiny")]
    VerifiedTiny,
}
