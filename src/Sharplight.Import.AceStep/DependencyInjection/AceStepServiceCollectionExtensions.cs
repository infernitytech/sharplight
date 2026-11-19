using Sharplight.Import.AceStep;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the ACE-Step checkpoint format.</summary>
public static class AceStepServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the ACE-Step importer (<see cref="AceStepImporter"/>), so the model loader opens ACE-Step 1.5 checkpoint folders (a diffusion transformer folder with its
        /// <c>config.json</c>, <c>model.safetensors</c> and <c>silence_latent.pt</c>, a <c>vae</c> folder and a Qwen3 text encoder folder). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddAceStepImporter() => services.AddModelImporter<AceStepImporter>();
    }
}
