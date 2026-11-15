using Microsoft.Extensions.Options;
using Sharplight.Import.HuggingFace;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the Hugging Face checkpoint format (safetensors directories).</summary>
public static class HuggingFaceServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the Hugging Face importer (<see cref="HuggingFaceImporter"/>), so the model loader opens checkpoint directories, with
        /// <see cref="HuggingFaceImportOptions"/> from Microsoft.Extensions.Options. Idempotent; each call adds <paramref name="configure"/>.
        /// </summary>
        /// <param name="configure">Configures the import options, or <see langword="null"/> for the defaults.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddHuggingFaceImporter(Action<HuggingFaceImportOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            var options = services.AddOptions<HuggingFaceImportOptions>();
            if (configure is not null)
            {
                options.Configure(configure);
            }

            return services.AddModelImporter(provider => new HuggingFaceImporter(provider.GetRequiredService<IOptions<HuggingFaceImportOptions>>().Value));
        }
    }
}
