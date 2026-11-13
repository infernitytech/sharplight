using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>Creates weight sources for byte ranges of the file being parsed.</summary>
internal interface IOnnxByteSliceProvider
{
    public IWeightSource Slice(long offset, long length);
}
