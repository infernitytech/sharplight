namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// The app's composition root for the engine: registers the execution providers (Vulkan, CPU), the built-in architecture adapters
/// and the model importers (GGUF, Hugging Face, ONNX, <c>.slm</c>) once through <see cref="SharplightDefaults.Register"/>. Everything
/// else talks to them through the registries.
/// </summary>
internal static class EngineBootstrap
{
    /// <summary>Registers providers, adapters and importers (idempotent).</summary>
    public static void Register() => SharplightDefaults.Register();
}
