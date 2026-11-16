using Sharplight.Import.Kokoro;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the Kokoro checkpoint format.</summary>
public static class KokoroServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the Kokoro importer (<see cref="KokoroImporter"/>), so the model loader opens Kokoro checkpoint directories (<c>config.json</c> with the
        /// <c>istftnet</c> and <c>plbert</c> sections beside the <c>.pth</c> weights and the <c>voices</c> folder). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddKokoroImporter() => services.AddModelImporter<KokoroImporter>();
    }
}
