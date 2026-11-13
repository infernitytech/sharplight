using System.Reflection;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>
/// Device profiles that ship with the tool (embedded <c>Profiles/&lt;name&gt;.json</c>), so <c>precompile --profile &lt;name&gt;</c>
/// works without exporting a profile first. Each was exported from a real device with <c>profile</c> (see the README).
/// </summary>
internal static class WellKnownProfiles
{
    private const string ResourcePrefix = "Profiles.";
    private const string ResourceSuffix = ".json";

    private static readonly Assembly s_assembly = typeof(WellKnownProfiles).Assembly;

    /// <summary>Names of the shipped profiles, sorted.</summary>
    public static IReadOnlyList<string> Names { get; } = [.. s_assembly.GetManifestResourceNames()
        .Where(n => n.StartsWith(ResourcePrefix, StringComparison.Ordinal) && n.EndsWith(ResourceSuffix, StringComparison.Ordinal))
        .Select(n => n[ResourcePrefix.Length..^ResourceSuffix.Length])
        .Order(StringComparer.Ordinal)];

    /// <summary>The shipped profile <paramref name="name"/>.</summary>
    public static DeviceProfile Get(string name)
    {
        using var stream = s_assembly.GetManifestResourceStream(ResourcePrefix + name + ResourceSuffix)
            ?? throw new SampleException($"Unknown device profile '{name}'; well-known profiles: {string.Join(", ", Names)}.", ExitCodes.InvalidInput);
        using var reader = new StreamReader(stream);
        return DeviceProfile.FromJson(reader.ReadToEnd());
    }

    /// <summary>Reads <paramref name="fileOrName"/> as a profile file when it exists, else as a well-known profile name.</summary>
    /// <exception cref="SampleException">Neither a readable profile file nor a well-known name.</exception>
    public static DeviceProfile Resolve(string fileOrName)
    {
        if (!File.Exists(fileOrName))
        {
            return Get(fileOrName);
        }

        try
        {
            return DeviceProfile.Load(fileOrName);
        }
        catch (InvalidDataException error)
        {
            throw new SampleException($"{fileOrName}: {error.Message}", ExitCodes.InvalidInput);
        }
    }
}
