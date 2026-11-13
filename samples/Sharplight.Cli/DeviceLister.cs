using Sharplight.Runtime;

namespace Sharplight.Cli;

/// <summary>Prints every registered provider with its availability, devices, features and capabilities.</summary>
internal static class DeviceLister
{
    public static void Print(TextWriter output)
    {
        foreach (var (provider, availability) in ProviderRegistry.Discover())
        {
            var state = availability.IsAvailable ? "available" : "unavailable";
            output.WriteLine($"{provider.Name}: {state} — {availability.Reason}");
            if (!availability.IsAvailable)
            {
                continue;
            }

            foreach (var device in provider.GetDevices())
            {
                PrintDevice(output, device);
            }
        }
    }

    private static void PrintDevice(TextWriter output, DeviceInfo device)
    {
        var capabilities = string.Join(", ", device.Capabilities.Select(pair => $"{pair.Key}={pair.Value}"));
        output.WriteLine($"  [{device.Index}] {device}");
        output.WriteLine($"      features: {string.Join(" ", device.Features.Order())}");
        output.WriteLine($"      {capabilities}");
    }
}
