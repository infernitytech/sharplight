using Sharplight.Import.Onnx;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the ONNX model format.</summary>
public static class OnnxServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the ONNX importer (<see cref="OnnxImporter"/>) and the generic graph adapter that translates ONNX graphs into Sharplight
        /// graphs (<see cref="OnnxGraphAdapter"/>). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddOnnxImporter() => services.AddModelImporter<OnnxImporter>().AddModelAdapter<OnnxGraphAdapter>();
    }
}
