using Microsoft.Extensions.DependencyInjection;
using Sharplight.Architectures;
using Sharplight.Cpu;
using Sharplight.Import.Diffusers;
using Sharplight.Import.Gguf;
using Sharplight.Import.HuggingFace;
using Sharplight.Import.Onnx;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vulkan;

namespace Sharplight;

/// <summary>
/// The one registration entry point for hosts that want Sharplight with its defaults: <c>services.AddSharplight()</c> wires up the
/// model formats, the execution providers and the task layer (<see cref="TaskServiceCollectionExtensions.AddSharplightTasks"/>);
/// task clients for concrete models are added on top (<c>services.AddTextGenerator(...)</c>, <c>services.AddSpeechRecognizer(...)</c>, ...).
/// </summary>
/// <remarks>
/// The importers, architecture adapters and execution providers still keep process-wide registries
/// (<see cref="Models.ModelImporterRegistry"/>, <see cref="Models.ModelAdapterRegistry"/>, <see cref="ProviderRegistry"/>); until they
/// become services, <see cref="AddSharplightModelFormats"/> and <see cref="AddSharplightProviders"/> fill them once per process when
/// called (idempotent, thread-safe), so loaders that consult them (<c>LlmModel.Load</c>, <c>TaskModel.Load</c>) work right away.
/// </remarks>
public static class SharplightServiceCollectionExtensions
{
    private static int s_formats;
    private static int s_providers;

    extension(IServiceCollection services)
    {
        /// <summary>Adds the model formats, the execution providers and the task layer.</summary>
        public IServiceCollection AddSharplight()
        {
            ArgumentNullException.ThrowIfNull(services);
            return services.AddSharplightModelFormats().AddSharplightProviders().AddSharplightTasks();
        }

        /// <summary>
        /// Adds the built-in architecture adapters (<see cref="BuiltinAdapters.RegisterAll"/>) and the GGUF, Hugging Face, ONNX and
        /// <c>.slm</c> package importers (the ONNX importer also brings its generic graph adapter).
        /// </summary>
        public IServiceCollection AddSharplightModelFormats()
        {
            ArgumentNullException.ThrowIfNull(services);
            if (Interlocked.Exchange(ref s_formats, 1) == 0)
            {
                BuiltinAdapters.RegisterAll();
                GgufImporter.Register();
                HuggingFaceImporter.Register();
                OnnxImporter.Register();
                DiffusersImporter.Register();
                ModelPackageImporter.Register();
            }

            return services;
        }

        /// <summary>
        /// Adds the Vulkan and CPU execution providers, in that order (so <see cref="ProviderRegistry.Discover"/> lists the GPU first).
        /// A provider whose driver is missing reports itself unavailable.
        /// </summary>
        public IServiceCollection AddSharplightProviders()
        {
            ArgumentNullException.ThrowIfNull(services);
            if (Interlocked.Exchange(ref s_providers, 1) == 0)
            {
                ProviderRegistry.Register(() => new VulkanProvider());
                CpuExecutionProvider.Register();
            }

            return services;
        }
    }
}
