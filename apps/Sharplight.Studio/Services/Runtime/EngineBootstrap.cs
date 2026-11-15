using Microsoft.Extensions.DependencyInjection;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// The app's composition root for the engine: one service provider built with <c>services.AddSharplight()</c>
/// (<see cref="SharplightDefaults.CreateServiceProvider"/>), which registers the execution providers (Vulkan, CPU), the built-in
/// architecture adapters, the model importers (GGUF, Hugging Face, ONNX, <c>.slm</c>) and the task layer.
/// </summary>
internal static class EngineBootstrap
{
    private static readonly Lazy<ServiceProvider> s_services = new(() => SharplightDefaults.CreateServiceProvider());

    /// <summary>The engine's services (task client provider, options), created on first use and kept for the app's lifetime.</summary>
    public static IServiceProvider Services => s_services.Value;

    /// <summary>Makes sure the engine services exist (idempotent).</summary>
    public static void Register() => _ = s_services.Value;
}
