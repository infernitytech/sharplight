using Sharplight.Import.Chatterbox;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the Chatterbox checkpoint format.</summary>
public static class ChatterboxServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the Chatterbox importer (<see cref="ChatterboxImporter"/>), so the model loader opens Chatterbox checkpoint directories
        /// (<c>t3_cfg.safetensors</c>, <c>s3gen.safetensors</c>, <c>tokenizer.json</c>, optionally <c>ve.safetensors</c> and <c>conds.pt</c>). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddChatterboxImporter() => services.AddModelImporter<ChatterboxImporter>();
    }
}
