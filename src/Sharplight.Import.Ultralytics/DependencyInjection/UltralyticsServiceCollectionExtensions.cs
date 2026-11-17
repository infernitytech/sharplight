using Sharplight.Import.Ultralytics;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the Ultralytics YOLO export format.</summary>
public static class UltralyticsServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the Ultralytics importer (<see cref="UltralyticsImporter"/>) and its adapter, so that the model loader opens the ONNX exports of YOLOv8, YOLO11 and YOLOv10 as object detection
        /// models. Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddUltralyticsImporter() => services.AddModelImporter<UltralyticsImporter>().AddModelAdapter<UltralyticsAdapter>();
    }
}
