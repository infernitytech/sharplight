using Sharplight.Models;

namespace Sharplight.Onnx;

internal sealed class MappedSliceProvider(MappedFile file) : IOnnxByteSliceProvider
{
    public IWeightSource Slice(long offset, long length) => new MappedRegionSource(file, offset, length);
}
