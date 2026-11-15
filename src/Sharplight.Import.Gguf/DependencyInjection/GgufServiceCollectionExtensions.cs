using Sharplight.Import.Gguf;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the GGUF model format.</summary>
public static class GgufServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>Adds the GGUF importer (<see cref="GgufImporter"/>), so the model loader opens <c>.gguf</c> files. Idempotent.</summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddGgufImporter() => services.AddModelImporter<GgufImporter>();
    }
}
