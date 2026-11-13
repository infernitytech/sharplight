using Sharplight.Models;

namespace Sharplight.Onnx;

internal sealed class ArraySliceProvider(byte[] data) : IOnnxByteSliceProvider
{
    public IWeightSource Slice(long offset, long length) => new ArrayWeightSource(data, (int)offset, (int)length);
}
