using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// The one registration entry point for hosts that want Sharplight with its defaults: <c>services.AddSharplight()</c> registers the
/// model formats, the built-in architectures, the execution providers and the task layer; task clients for concrete models are added on
/// top (<c>services.AddTextGenerator(...)</c>, <c>services.AddSpeechRecognizer(...)</c>, ...).
/// </summary>
/// <remarks>
/// Every registration is idempotent and only adds services, so hosts can call the narrower methods (or the per-package ones such as
/// <c>services.AddGgufImporter()</c>, <c>services.AddCpuProvider()</c>) instead, or add their own importers, adapters and providers next
/// to the defaults. Options are ordinary Microsoft.Extensions.Options: <c>services.Configure&lt;VulkanProviderOptions&gt;(...)</c>,
/// <c>services.Configure&lt;HuggingFaceImportOptions&gt;(...)</c>. Nothing is process-wide: two containers with different registrations
/// load models independently.
/// </remarks>
public static class SharplightServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the model formats and architectures (<see cref="AddSharplightModelFormats"/>), the execution providers
        /// (<see cref="AddSharplightProviders"/>), the task layer (<c>services.AddSharplightTasks()</c>) and the native audio services
        /// (<c>services.AddSharplightAudio()</c>: decoding, encoding, playback and capture), and the storage layout
        /// (<c>services.AddSharplightStorage()</c>: where configuration, data, models and caches are kept). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplight()
        {
            ArgumentNullException.ThrowIfNull(services);
            return services.AddSharplightStorage().AddSharplightModelFormats().AddSharplightProviders().AddSharplightTasks().AddSharplightAudio();
        }

        /// <summary>
        /// Adds the GGUF, Hugging Face (safetensors and PyTorch shards), ONNX, whisper.cpp, diffusers and <c>.slm</c> package importers and the built-in architecture adapters, which the
        /// <see cref="ModelLoader"/> and the <see cref="ModelConverter"/> use. Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightModelFormats()
        {
            ArgumentNullException.ThrowIfNull(services);
            return services.AddBuiltinArchitectures()
                .AddGgufImporter()
                .AddPyTorchImporter()
                .AddHuggingFaceImporter()
                .AddOnnxImporter()
                .AddWhisperCppImporter()
                .AddKokoroImporter()
                .AddSnacImporter()
                .AddDiffusersImporter()
                .AddPackageImporter();
        }

        /// <summary>
        /// Adds the Vulkan and CPU execution providers, in that order (so <see cref="ExecutionProviderCatalog.Discover"/> lists the GPU
        /// first). A provider whose driver is missing reports itself unavailable. Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightProviders()
        {
            ArgumentNullException.ThrowIfNull(services);
            return services.AddVulkanProvider().AddCpuProvider();
        }
    }
}
