using Sharplight.Import.Snac;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the SNAC codec checkpoint format.</summary>
public static class SnacServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the SNAC importer (<see cref="SnacImporter"/>), so the model loader opens SNAC checkpoint directories (<c>config.json</c> with
        /// <c>vq_strides</c> beside <c>pytorch_model.bin</c>). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSnacImporter() => services.AddModelImporter<SnacImporter>();
    }
}
