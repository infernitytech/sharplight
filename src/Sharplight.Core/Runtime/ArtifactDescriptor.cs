using System.Security.Cryptography;
using System.Text;

namespace Sharplight.Runtime;

/// <summary>Helpers for artifact descriptors: string → string maps that describe an artifact set (see <see cref="ArtifactDescriptorKeys"/>).</summary>
public static class ArtifactDescriptor
{
    /// <summary>Reads a descriptor value; empty when the key is absent.</summary>
    public static string Get(IReadOnlyDictionary<string, string> descriptor, string key) => descriptor.TryGetValue(key, out var value) ? value : "";

    /// <summary>The descriptor as <c>key=value</c> pairs separated by spaces, <see cref="ArtifactDescriptorKeys.Provider"/>, tier and kind first, then the other keys in ordinal order.</summary>
    public static string Format(IReadOnlyDictionary<string, string> descriptor)
    {
        string[] leading = [ArtifactDescriptorKeys.Provider, ArtifactDescriptorKeys.Tier, ArtifactDescriptorKeys.Kind];
        var keys = leading.Where(descriptor.ContainsKey).Concat(descriptor.Keys.Where(k => !leading.Contains(k)).Order(StringComparer.Ordinal));
        return string.Join(" ", keys.Select(k => $"{k}={descriptor[k]}"));
    }

    /// <summary>
    /// Stable identifier of a descriptor: <c>provider.tier.kind.</c> followed by a hash of all its pairs. Equal descriptors have
    /// equal ids, so a set stored again for the same descriptor replaces the earlier one.
    /// </summary>
    public static string Id(IReadOnlyDictionary<string, string> descriptor)
    {
        var canonical = new StringBuilder();
        foreach (var key in descriptor.Keys.Order(StringComparer.Ordinal))
        {
            canonical.Append(key).Append('=').Append(descriptor[key]).Append('\n');
        }

        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString())))[..16].ToLowerInvariant();
        return $"{Get(descriptor, ArtifactDescriptorKeys.Provider)}.{Get(descriptor, ArtifactDescriptorKeys.Tier)}.{Get(descriptor, ArtifactDescriptorKeys.Kind)}.{hash}";
    }
}
