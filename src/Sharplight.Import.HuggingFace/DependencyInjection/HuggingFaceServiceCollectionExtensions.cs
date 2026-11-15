using Microsoft.Extensions.Options;
using Sharplight.Import.HuggingFace;
using Sharplight.Import.PyTorch;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the Hugging Face checkpoint format (safetensors directories).</summary>
public static class HuggingFaceServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the Hugging Face importer (<see cref="HuggingFaceImporter"/>), so the model loader opens checkpoint directories (safetensors or PyTorch <c>pytorch_model*.bin</c> shards, read with the <see cref="TorchCheckpointOptions"/> of <c>services.AddPyTorchImporter()</c>), with
        /// <see cref="HuggingFaceImportOptions"/> from Microsoft.Extensions.Options. Idempotent; each call adds <paramref name="configure"/>.
        /// </summary>
        /// <param name="configure">Configures the import options, or <see langword="null"/> for the defaults.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddHuggingFaceImporter(Action<HuggingFaceImportOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            services.AddPyTorchImporter();
            var options = services.AddOptions<HuggingFaceImportOptions>();
            options.PostConfigure<IOptions<TorchCheckpointOptions>>((hf, torch) => hf.PyTorch = torch.Value);

            if (configure is not null)
            {
                options.Configure(configure);
            }

            return services.AddModelImporter(provider => new HuggingFaceImporter(provider.GetRequiredService<IOptions<HuggingFaceImportOptions>>().Value));
        }
    }
}
