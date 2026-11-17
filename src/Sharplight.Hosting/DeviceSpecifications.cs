namespace Sharplight.Hosting;

/// <summary>
/// The device names of the hosting options: <c>auto</c>, <c>cpu</c>, a GPU index (<c>0</c>: the first available GPU provider's device),
/// a device id (<c>provider:index</c>, e.g. <c>vulkan:1</c>), or several of them separated by commas (the model is split over them in
/// pipeline order).
/// </summary>
public static class DeviceSpecifications
{
    /// <summary>The GPU with the most free memory (the CPU when there is no GPU).</summary>
    public const string Auto = "auto";

    /// <summary>The CPU.</summary>
    public const string Cpu = "cpu";

    /// <summary>Splits a device specification into its parts (trimmed, empty parts dropped; <see cref="Auto"/> when empty).</summary>
    /// <param name="specification">The specification, or <see langword="null"/>.</param>
    /// <returns>One entry per device.</returns>
    public static IReadOnlyList<string> Split(string? specification)
    {
        var parts = (specification ?? "").Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        return parts.Length == 0 ? [Auto] : parts;
    }

    /// <summary>True when <paramref name="specification"/> names exactly the CPU.</summary>
    /// <param name="specification">The specification.</param>
    public static bool IsCpu(string? specification)
    {
        var parts = Split(specification);
        return parts.Count == 1 && parts[0].Equals(Cpu, StringComparison.OrdinalIgnoreCase);
    }
}
