using Microsoft.Extensions.DependencyInjection;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// The engine services the app uses, resolved from the composition root's container (built with <c>services.AddSharplight()</c>:
/// the Vulkan and CPU providers, the built-in architectures, the GGUF, Hugging Face, ONNX, whisper.cpp, diffusers and <c>.slm</c> formats and the task layer).
/// </summary>
/// <param name="Services">The container (task client provider, options).</param>
/// <param name="Loader">Opens and builds models.</param>
/// <param name="Providers">The execution providers.</param>
/// <param name="Converter">Converts models to <c>.slm</c> packages.</param>
internal sealed record EngineServices(IServiceProvider Services, ModelLoader Loader, ExecutionProviderCatalog Providers, ModelConverter Converter)
{
    /// <summary>Resolves the engine services from a container.</summary>
    /// <param name="services">A container with <c>services.AddSharplight()</c>.</param>
    public static EngineServices From(IServiceProvider services)
        => new(services, services.GetRequiredService<ModelLoader>(), services.GetRequiredService<ExecutionProviderCatalog>(),
            services.GetRequiredService<ModelConverter>());
}
