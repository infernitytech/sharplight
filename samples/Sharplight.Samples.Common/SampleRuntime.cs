using Microsoft.Extensions.DependencyInjection;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.Samples;

/// <summary>
/// The Sharplight services a sample uses: a container with <c>services.AddSharplight()</c> (the GGUF, Hugging Face, ONNX and package
/// formats, the built-in architectures, the Vulkan and CPU providers and the task layer). Create one in <c>Program.cs</c> and pass it
/// to the steps that load or convert models; dispose it at exit.
/// </summary>
public sealed class SampleRuntime : IDisposable
{
    private const string CpuProviderName = "cpu";
    private readonly ServiceProvider _services;

    /// <summary>Builds the container.</summary>
    /// <param name="configure">Adds further services, or <see langword="null"/>.</param>
    public SampleRuntime(Action<IServiceCollection>? configure = null)
    {
        _services = SharplightDefaults.CreateServiceProvider(configure);
    }

    /// <summary>The container.</summary>
    public IServiceProvider Services => _services;

    /// <summary>Opens and builds models.</summary>
    public ModelLoader Loader => _services.GetRequiredService<ModelLoader>();

    /// <summary>The execution providers (Vulkan, then CPU).</summary>
    public ExecutionProviderCatalog Providers => _services.GetRequiredService<ExecutionProviderCatalog>();

    /// <summary>Converts models to <c>.slm</c> packages.</summary>
    public ModelConverter Converter => _services.GetRequiredService<ModelConverter>();

    /// <summary>Returns the first available provider: the CPU provider when <paramref name="cpu"/> is set, otherwise a GPU provider.</summary>
    /// <param name="cpu">Select the CPU provider.</param>
    /// <returns>The provider; owned by the runtime.</returns>
    /// <exception cref="SampleException">No matching provider is available on this machine.</exception>
    public IExecutionProvider SelectProvider(bool cpu)
    {
        foreach (var (provider, availability) in Providers.Discover())
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
    /// <param name="provider">The provider.</param>
    /// <param name="index">The device index.</param>
    /// <returns>The device description.</returns>
    /// <exception cref="SampleException">The index is out of range.</exception>
    public static DeviceInfo GetDevice(IExecutionProvider provider, int index)
    {
        ArgumentNullException.ThrowIfNull(provider);
        var devices = provider.GetDevices();
        if (index < 0 || index >= devices.Count)
        {
            throw new SampleException($"Device index {index} is out of range: provider '{provider.Name}' has {devices.Count} device(s).");
        }

        return devices[index];
    }

    /// <summary>Disposes the container (and the providers it created).</summary>
    public void Dispose() => _services.Dispose();
}
