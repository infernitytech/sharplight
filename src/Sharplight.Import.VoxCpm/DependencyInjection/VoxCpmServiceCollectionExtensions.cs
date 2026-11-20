using Sharplight.Import.VoxCpm;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the VoxCPM checkpoint format.</summary>
public static class VoxCpmServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the VoxCPM importer (<see cref="VoxCpmImporter"/>), so the model loader opens VoxCPM2 checkpoint directories (<c>config.json</c> with
        /// <c>"architecture": "voxcpm2"</c>, <c>model.safetensors</c>, <c>audiovae.pth</c> or <c>audiovae.safetensors</c>, <c>tokenizer.json</c>). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddVoxCpmImporter() => services.AddModelImporter<VoxCpmImporter>();
    }
}
