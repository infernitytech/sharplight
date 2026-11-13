using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Samples;

/// <summary>Registers the execution providers and model importers every LLM sample uses.</summary>
public static class SampleRuntime
{
    private const string CpuProviderName = "cpu";

    /// <summary>Registers the Vulkan and CPU providers and the GGUF, Hugging Face and model-package importers.</summary>
    public static void RegisterProvidersAndImporters()
    {
        ProviderRegistry.Register(() => new VulkanProvider());
        CpuExecutionProvider.Register();
        GgufImporter.Register();
        HuggingFaceImporter.Register();
        ModelPackageImporter.Register();
    }

    /// <summary>Returns the first available provider: the CPU provider when <paramref name="cpu"/> is set, otherwise a GPU provider.</summary>
    /// <exception cref="SampleException">No matching provider is available on this machine.</exception>
    public static IExecutionProvider SelectProvider(bool cpu)
    {
        foreach (var (provider, availability) in ProviderRegistry.Discover())
        {
            bool isCpu = provider.Name == CpuProviderName;
            if (availability.IsAvailable && isCpu == cpu)
            {
                return provider;
            }
        }

        throw new SampleException(cpu ? "The CPU provider is not available." : "No GPU provider is available; use --cpu to run on the CPU.");
    }

    /// <summary>Returns the device at <paramref name="index"/> of <paramref name="provider"/>.</summary>
    /// <exception cref="SampleException">The index is out of range.</exception>
    public static DeviceInfo GetDevice(IExecutionProvider provider, int index)
    {
        var devices = provider.GetDevices();
        if (index < 0 || index >= devices.Count)
        {
            throw new SampleException($"Device index {index} is out of range: provider '{provider.Name}' has {devices.Count} device(s).");
        }

        return devices[index];
    }
}
