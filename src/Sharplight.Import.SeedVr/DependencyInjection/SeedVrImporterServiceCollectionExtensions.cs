using Sharplight.Import.SeedVr;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the SeedVR2 checkpoint format.</summary>
public static class SeedVrImporterServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the SeedVR2 importer (<see cref="SeedVrImporter"/>), so the model loader opens SeedVR2 model folders (the DiT checkpoint,
        /// <c>ema_vae.pth</c> and <c>pos_emb.pt</c>). The architecture comes from <c>services.AddSeedVrArchitecture()</c>. Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSeedVrImporter() => services.AddModelImporter<SeedVrImporter>();
    }
}
