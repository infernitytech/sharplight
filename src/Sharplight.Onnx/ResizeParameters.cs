namespace Sharplight.Onnx;

/// <summary>The interpolation attributes of an ONNX Resize/Upsample node (see <see cref="ResizeAxis"/>).</summary>
/// <param name="Mode">"nearest", "linear" or "cubic".</param>
/// <param name="CoordinateMode">coordinate_transformation_mode.</param>
/// <param name="NearestMode">Rounding of nearest mode.</param>
/// <param name="CubicCoeffA">Cubic kernel parameter.</param>
/// <param name="ExcludeOutside">Zero (and renormalize) weights of samples outside the input.</param>
/// <param name="Antialias">Widen linear/cubic kernels when downsampling.</param>
/// <param name="ExtrapolationValue">Value of tf_crop_and_resize outputs whose source lies outside the input.</param>
internal sealed record ResizeParameters(string Mode, string CoordinateMode, string NearestMode, double CubicCoeffA, bool ExcludeOutside, bool Antialias,
    double ExtrapolationValue);
