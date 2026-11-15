using Sharplight.Architectures;
using Sharplight.Cpu;
using Sharplight.Import.Gguf;
using Sharplight.Import.HuggingFace;
using Sharplight.Import.Onnx;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight;

/// <summary>
/// The composition root for hosts that want Sharplight's defaults. The core (<c>Sharplight.Core</c>) only consults
/// <see cref="Models.ModelImporterRegistry"/>, <see cref="Models.ModelAdapterRegistry"/> and <see cref="ProviderRegistry"/>; this type
/// fills them with everything that ships with Sharplight, so a host is one call away from <c>LlmModel.Load</c> or <c>TaskModel.Load</c>.
/// Every method is idempotent and thread-safe; hosts that want a narrower setup call the individual <c>Register</c> methods of the
/// importer, adapter and provider assemblies instead.
/// </summary>
/// <example>
/// <code>
/// SharplightDefaults.Register();
/// using var model = LlmModel.Load("qwen3-4b-q4_k_m.gguf", new LlmLoadOptions { Devices = [slot] });
/// </code>
/// </example>
public static class SharplightDefaults
{
    private static int s_formats;
    private static int s_providers;

    /// <summary>Registers the model formats (<see cref="RegisterModelFormats"/>) and the execution providers (<see cref="RegisterProviders"/>).</summary>
    public static void Register()
    {
        RegisterModelFormats();
        RegisterProviders();
    }

    /// <summary>
    /// Registers the built-in architecture adapters (<see cref="BuiltinAdapters.RegisterAll"/>) and the GGUF, Hugging Face, ONNX and
    /// <c>.slm</c> package importers (the ONNX importer also registers its generic graph adapter).
    /// </summary>
    public static void RegisterModelFormats()
    {
        if (Interlocked.Exchange(ref s_formats, 1) == 1)
        {
            return;
        }

        BuiltinAdapters.RegisterAll();
        GgufImporter.Register();
        HuggingFaceImporter.Register();
        OnnxImporter.Register();
        ModelPackageImporter.Register();
    }

    /// <summary>
    /// Registers the Vulkan and CPU execution providers with <see cref="ProviderRegistry"/>, in that order (so
    /// <see cref="ProviderRegistry.Discover"/> lists the GPU first). A provider whose driver is missing reports itself unavailable.
    /// </summary>
    public static void RegisterProviders()
    {
        if (Interlocked.Exchange(ref s_providers, 1) == 1)
        {
            return;
        }

        ProviderRegistry.Register(() => new VulkanProvider());
        CpuExecutionProvider.Register();
    }
}
