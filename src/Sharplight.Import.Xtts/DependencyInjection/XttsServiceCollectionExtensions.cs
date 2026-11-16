using Sharplight.Import.Xtts;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the XTTS v2 checkpoint format.</summary>
public static class XttsServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the XTTS importer (<see cref="XttsImporter"/>), so the model loader opens XTTS v2 checkpoint folders (<c>config.json</c>, <c>model.pth</c>,
        /// <c>vocab.json</c>, optionally <c>speakers_xtts.pth</c>). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddXttsImporter() => services.AddModelImporter<XttsImporter>();
    }
}
