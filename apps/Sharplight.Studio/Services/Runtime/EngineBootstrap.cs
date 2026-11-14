using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Onnx;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// The app's composition root for the engine: registers the execution providers (Vulkan, CPU) and the model importers (GGUF,
/// Hugging Face, ONNX, <c>.slm</c>) once. Everything else talks to them through the provider registry.
/// </summary>
internal static class EngineBootstrap
{
    private static int s_registered;

    /// <summary>Registers providers and importers (idempotent).</summary>
    public static void Register()
    {
        if (Interlocked.Exchange(ref s_registered, 1) != 0)
        {
            return;
        }

        ProviderRegistry.Register(() => new VulkanProvider());
        CpuExecutionProvider.Register();
        GgufImporter.Register();
        HuggingFaceImporter.Register();
        OnnxImporter.Register();
        ModelPackageImporter.Register();
    }
}
