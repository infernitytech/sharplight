namespace Sharplight.Onnx;

/// <summary>The axes and directions of an ONNX Scan: per scan input and per scan output (direction 1 = reverse).</summary>
internal sealed record ScanSpec(long[] InputAxes, long[] InputDirections, long[] OutputAxes, long[] OutputDirections);
