using System.Globalization;

namespace Sharplight.Runtime;

/// <summary>
/// Caps a device's memory below what the hardware offers, the same on every provider: a smaller device heap, less (or no)
/// device memory the host can map directly, a smaller largest allocation, less page-locked host memory. A provider applies the
/// smaller of each cap and the hardware's own figure, and behaves as on hardware with those figures: it falls back where hardware
/// with that limit would (other memory types, smaller staging buffers, pageable copies) and fails with a
/// <see cref="DeviceOutOfMemoryException"/> where it would. Useful to leave memory to other applications, and to test the fallbacks
/// of small or unusual hardware on a large GPU.
/// </summary>
/// <remarks>
/// Process-wide defaults come from the environment (sizes in bytes, or with a K, M, G or T suffix for binary multiples):
/// <list type="bullet">
/// <item><c>SHARPLIGHT_LIMIT_DEVICE_MEMORY</c>: <see cref="DeviceMemory"/>.</item>
/// <item><c>SHARPLIGHT_LIMIT_HOST_VISIBLE_DEVICE_MEMORY</c>: <see cref="HostVisibleDeviceMemory"/> (0 hides the type).</item>
/// <item><c>SHARPLIGHT_LIMIT_MAX_ALLOCATION</c>: <see cref="MaxAllocation"/>.</item>
/// <item><c>SHARPLIGHT_LIMIT_PINNED_HOST_MEMORY</c>: <see cref="PinnedHostMemory"/> (0 disables page-locked staging).</item>
/// </list>
/// A cap in <see cref="DeviceOptions.MemoryLimits"/> and one from the environment combine to the smaller of both.
/// </remarks>
public sealed class DeviceMemoryLimits
{
    /// <summary>Environment variable of <see cref="DeviceMemory"/>.</summary>
    public const string DeviceMemoryVariable = "SHARPLIGHT_LIMIT_DEVICE_MEMORY";
    /// <summary>Environment variable of <see cref="HostVisibleDeviceMemory"/>.</summary>
    public const string HostVisibleDeviceMemoryVariable = "SHARPLIGHT_LIMIT_HOST_VISIBLE_DEVICE_MEMORY";
    /// <summary>Environment variable of <see cref="MaxAllocation"/>.</summary>
    public const string MaxAllocationVariable = "SHARPLIGHT_LIMIT_MAX_ALLOCATION";
    /// <summary>Environment variable of <see cref="PinnedHostMemory"/>.</summary>
    public const string PinnedHostMemoryVariable = "SHARPLIGHT_LIMIT_PINNED_HOST_MEMORY";

    /// <summary>
    /// Bytes of device-local memory the device may hold in total (weights, state, activations; null = the hardware's). Unlike
    /// <see cref="DeviceOptions.MemoryBudget"/>, which only steers placement, an allocation beyond this cap fails as on a device with
    /// that much memory, and the device reports it as its total.
    /// </summary>
    public long? DeviceMemory { get; set; }

    /// <summary>
    /// Bytes of device-local memory the host can map (the PCIe BAR window of a discrete GPU; null = the hardware's). 0 hides that
    /// memory, as on GPUs without it: staging then goes through host memory. Devices whose memory is all host-visible (integrated
    /// GPUs, CPU implementations) have no separate window and ignore this cap.
    /// </summary>
    public long? HostVisibleDeviceMemory { get; set; }

    /// <summary>Largest single device allocation in bytes (null = the driver's limit); larger buffers are split or fail as on such hardware.</summary>
    public long? MaxAllocation { get; set; }

    /// <summary>Bytes of page-locked (pinned) host memory for staging (null = what the system grants); 0 copies from pageable memory.</summary>
    public long? PinnedHostMemory { get; set; }

    /// <summary>True when no cap is set.</summary>
    public bool IsEmpty => DeviceMemory is null && HostVisibleDeviceMemory is null && MaxAllocation is null && PinnedHostMemory is null;

    /// <summary>The caps set in the environment (see the remarks); empty when none is.</summary>
    /// <exception cref="FormatException">A variable is set but is not a size.</exception>
    public static DeviceMemoryLimits FromEnvironment() => new()
    {
        DeviceMemory = ReadVariable(DeviceMemoryVariable),
        HostVisibleDeviceMemory = ReadVariable(HostVisibleDeviceMemoryVariable),
        MaxAllocation = ReadVariable(MaxAllocationVariable),
        PinnedHostMemory = ReadVariable(PinnedHostMemoryVariable),
    };

    /// <summary>The caps that apply to a device opened with <paramref name="options"/>: its own and the environment's, the smaller of each.</summary>
    /// <param name="options">The device options.</param>
    public static DeviceMemoryLimits For(DeviceOptions options)
    {
        var environment = FromEnvironment();
        var own = options.MemoryLimits;
        return new DeviceMemoryLimits
        {
            DeviceMemory = Smaller(own.DeviceMemory, environment.DeviceMemory),
            HostVisibleDeviceMemory = Smaller(own.HostVisibleDeviceMemory, environment.HostVisibleDeviceMemory),
            MaxAllocation = Smaller(own.MaxAllocation, environment.MaxAllocation),
            PinnedHostMemory = Smaller(own.PinnedHostMemory, environment.PinnedHostMemory),
        };
    }

    /// <summary>The smaller of a hardware figure and an optional cap.</summary>
    /// <param name="hardware">The hardware's figure.</param>
    /// <param name="cap">The cap, or null.</param>
    public static long Apply(long hardware, long? cap) => cap is { } limit ? Math.Min(hardware, Math.Max(0, limit)) : hardware;

    /// <summary>Parses a size: bytes, or a number with a K, M, G or T suffix (binary multiples, an optional trailing B or iB).</summary>
    /// <param name="text">The text, e.g. <c>256M</c>, <c>1.5GiB</c> or <c>1048576</c>.</param>
    /// <exception cref="FormatException"><paramref name="text"/> is not a size.</exception>
    public static long ParseSize(string text)
    {
        string value = text.Trim();
        if (value.EndsWith("iB", StringComparison.OrdinalIgnoreCase))
        {
            value = value[..^2];
        }
        else if (value.Length > 1 && (value.EndsWith('B') || value.EndsWith('b')))
        {
            value = value[..^1];
        }

        long multiplier = 1;
        if (value.Length > 0)
        {
            int shift = char.ToUpperInvariant(value[^1]) switch
            {
                'K' => 10,
                'M' => 20,
                'G' => 30,
                'T' => 40,
                _ => 0,
            };
            if (shift > 0)
            {
                multiplier = 1L << shift;
                value = value[..^1];
            }
        }

        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double number) || number < 0 || double.IsInfinity(number))
        {
            throw new FormatException($"'{text}' is not a memory size (bytes, or a number with a K, M, G or T suffix).");
        }

        return (long)Math.Round(number * multiplier);
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var parts = new List<string>();
        void Add(string name, long? value)
        {
            if (value is { } bytes)
            {
                parts.Add($"{name} {bytes / 1048576.0:0.#} MiB");
            }
        }

        Add("device memory", DeviceMemory);
        Add("host-visible device memory", HostVisibleDeviceMemory);
        Add("max allocation", MaxAllocation);
        Add("pinned host memory", PinnedHostMemory);
        return parts.Count == 0 ? "no limits" : string.Join(", ", parts);
    }

    private static long? Smaller(long? a, long? b) => a is null ? b : b is null ? a : Math.Min(a.Value, b.Value);

    private static long? ReadVariable(string name)
    {
        string? text = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        try
        {
            return ParseSize(text);
        }
        catch (FormatException error)
        {
            throw new FormatException($"{name}: {error.Message}", error);
        }
    }
}
