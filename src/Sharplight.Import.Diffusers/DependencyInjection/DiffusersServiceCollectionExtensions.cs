using Sharplight.Import.Diffusers;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the diffusion checkpoint formats (diffusers folders and single-file Stable Diffusion checkpoints).</summary>
public static class DiffusersServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the diffusers importer (<see cref="DiffusersImporter"/>), so the model loader opens diffusers folders and single-file
        /// Stable Diffusion checkpoints; it ranks above the generic safetensors and GGUF importers for the files it claims. Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddDiffusersImporter() => services.AddModelImporter<DiffusersImporter>();
    }
}
