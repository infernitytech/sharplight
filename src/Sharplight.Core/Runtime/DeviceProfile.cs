using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace Sharplight.Runtime;

/// <summary>
/// What code generation for a device depends on, captured so code can be produced for a device that is not present: the
/// provider and its code target, subgroup size, shared memory, the features kernels may use and the submission budget, plus
/// display information and the memory placement decisions are based on. Exported from an opened device
/// (<see cref="IComputeDevice.CreateProfile"/>) and opened for offline compilation through an
/// <see cref="IOfflineCompilationProvider"/>. Serialized as JSON (<see cref="ToJson"/>, <see cref="FromJson"/>).
/// </summary>
public sealed class DeviceProfile
{
    /// <summary>Version of the JSON layout this type writes.</summary>
    public const int CurrentFormatVersion = 1;

    /// <summary>Version of the JSON layout (<see cref="CurrentFormatVersion"/> for profiles written by this version).</summary>
    public int FormatVersion { get; init; } = CurrentFormatVersion;

    /// <summary>Execution provider the profile is for (<see cref="DeviceInfo.Provider"/>, e.g. <c>vulkan</c>).</summary>
    public required string Provider { get; init; }

    /// <summary>Device name for display (<see cref="DeviceInfo.Name"/>).</summary>
    public required string Name { get; init; }

    /// <summary>Vendor name for display (<see cref="DeviceInfo.Vendor"/>); empty when unknown.</summary>
    public string Vendor { get; init; } = "";

    /// <summary>Device class; decides host-side choices that shape the graphs, such as whether sampling runs on the device.</summary>
    public DeviceKind Kind { get; init; } = DeviceKind.DiscreteGpu;

    /// <summary>Device-local memory in bytes used for placement; 0 = unknown (an offline device then assumes plenty).</summary>
    public long TotalMemory { get; init; }

    /// <summary>Provider code generation target (Vulkan: the SPIR-V profile, e.g. <c>spirv_1_6</c>).</summary>
    public required string CodeTarget { get; init; }

    /// <summary>Lanes per subgroup kernels are specialized for.</summary>
    public required int SubgroupSize { get; init; }

    /// <summary>Shared (workgroup) memory per workgroup in bytes.</summary>
    public required long MaxSharedMemory { get; init; }

    /// <summary>Device features (<see cref="DeviceFeatures"/> names) the device may use, ordinally sorted.</summary>
    public IReadOnlyList<string> Features { get; init; } = [];

    /// <summary>See <see cref="DeviceInfo.SubmissionBudgetMs"/>; 0 = no limit.</summary>
    public double SubmissionBudgetMs { get; init; }

    /// <summary>Provider-specific informational details (vendor and device ids, API version, limits); not used for compilation.</summary>
    public IReadOnlyDictionary<string, string> Properties { get; init; } = new Dictionary<string, string>();

    /// <summary>A copy with <paramref name="disabled"/> removed from <see cref="Features"/> (e.g. a profile without matrix units).</summary>
    /// <param name="disabled">Feature names to remove.</param>
    /// <param name="name">Display name of the copy, or <see langword="null"/> to keep <see cref="Name"/>.</param>
    public DeviceProfile WithoutFeatures(IEnumerable<string> disabled, string? name = null)
    {
        var removed = new HashSet<string>(disabled, StringComparer.OrdinalIgnoreCase);
        return new DeviceProfile
        {
            FormatVersion = FormatVersion,
            Provider = Provider,
            Name = name ?? Name,
            Vendor = Vendor,
            Kind = Kind,
            TotalMemory = TotalMemory,
            CodeTarget = CodeTarget,
            SubgroupSize = SubgroupSize,
            MaxSharedMemory = MaxSharedMemory,
            Features = [.. Features.Where(f => !removed.Contains(f))],
            SubmissionBudgetMs = SubmissionBudgetMs,
            Properties = Properties,
        };
    }

    /// <summary>The profile as indented JSON.</summary>
    public string ToJson() => JsonSerializer.Serialize(this, (JsonTypeInfo<DeviceProfile>)s_writeOptions.GetTypeInfo(typeof(DeviceProfile)));

    /// <summary>Source-generated metadata, writing characters such as <c>+</c> and <c>></c> unescaped (the file is read by people, not embedded in HTML).</summary>
    private static readonly JsonSerializerOptions s_writeOptions = new(DeviceProfileJsonContext.Default.Options) { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    /// <summary>Reads a profile written by <see cref="ToJson"/>.</summary>
    /// <param name="json">The JSON text.</param>
    /// <exception cref="InvalidDataException">The text is not a valid profile or was written by a newer version.</exception>
    public static DeviceProfile FromJson(string json)
    {
        DeviceProfile? profile;
        try
        {
            profile = JsonSerializer.Deserialize(json, DeviceProfileJsonContext.Default.DeviceProfile);
        }
        catch (JsonException error)
        {
            throw new InvalidDataException($"Not a valid device profile: {error.Message}", error);
        }

        if (profile is null)
        {
            throw new InvalidDataException("Not a valid device profile: empty document.");
        }

        if (profile.FormatVersion is < 1 or > CurrentFormatVersion)
        {
            throw new InvalidDataException($"Device profile format {profile.FormatVersion} is not supported (this version reads 1 to {CurrentFormatVersion}).");
        }

        if (profile.SubgroupSize <= 0 || profile.MaxSharedMemory <= 0 || profile.Provider.Length == 0 || profile.CodeTarget.Length == 0)
        {
            throw new InvalidDataException("Not a valid device profile: provider, code target, subgroup size and shared memory are required.");
        }

        return profile;
    }

    /// <summary>Reads a profile file (<see cref="FromJson"/>).</summary>
    /// <param name="path">The JSON file.</param>
    /// <exception cref="InvalidDataException">The file is not a valid profile.</exception>
    public static DeviceProfile Load(string path) => FromJson(File.ReadAllText(path));

    /// <inheritdoc/>
    public override string ToString()
        => $"{Provider} profile '{Name}' ({CodeTarget}, subgroup {SubgroupSize}, shared {MaxSharedMemory} B, features {string.Join(",", Features)})";
}
