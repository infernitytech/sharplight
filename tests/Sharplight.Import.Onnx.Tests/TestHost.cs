using Microsoft.Extensions.DependencyInjection;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.Import.Onnx.Tests;

/// <summary>
/// The services the tests load models through: the formats, architectures and execution providers of Sharplight
/// (<c>services.AddSharplight()</c>), built once per test assembly.
/// </summary>
internal static class TestHost
{
    private static readonly Lazy<ServiceProvider> s_services = new(() => SharplightDefaults.CreateServiceProvider());

    /// <summary>The container.</summary>
    public static IServiceProvider Services => s_services.Value;

    /// <summary>Opens and builds models.</summary>
    public static ModelLoader Loader => Services.GetRequiredService<ModelLoader>();

    /// <summary>The execution providers (Vulkan, then CPU).</summary>
    public static ExecutionProviderCatalog Providers => Services.GetRequiredService<ExecutionProviderCatalog>();

    /// <summary>Converts models to packages.</summary>
    public static ModelConverter Converter => Services.GetRequiredService<ModelConverter>();
}
