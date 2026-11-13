using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Package;

/// <summary>
/// Encodes the WIDX section: <c>i32 count</c>, then per weight its name, type name, <c>i32 rank</c>, <c>i64</c> dimensions,
/// <c>i64 offset</c> (absolute) and <c>i64 length</c>.
/// </summary>
internal static class WeightIndexCodec
{
    public static void Write(BinaryWriter w, List<WeightInfo> weights, long[] offsets)
    {
        w.Write(weights.Count);
        for (int i = 0; i < weights.Count; i++)
        {
            var wi = weights[i];
            MetadataCodec.WriteString(w, wi.Name);
            MetadataCodec.WriteString(w, wi.Type.Name);
            var dims = wi.Shape.Dims;
            w.Write(dims.Length);
            foreach (var d in dims)
            {
                w.Write(d.IsSymbolic ? throw new InvalidDataException($"Weight {wi.Name} has a symbolic shape.") : d.Fixed);
            }

            w.Write(offsets[i]);
            w.Write(wi.ByteSize);
        }
        w.Flush();
    }

    /// <summary>Reads and validates the index.</summary>
    /// <param name="r">Reader positioned at the start of the WIDX section.</param>
    /// <param name="fileLength">Length of the package; every weight must lie inside it.</param>
    /// <exception cref="InvalidDataException">A weight lies outside the file or its size does not match its type and shape.</exception>
    public static List<PackageWeightInfo> Read(BinaryReader r, long fileLength)
    {
        int n = r.ReadInt32();
        var weights = new List<PackageWeightInfo>(n);
        for (int i = 0; i < n; i++)
        {
            string name = MetadataCodec.ReadString(r);
            var type = DataType.Get(MetadataCodec.ReadString(r));
            int rank = r.ReadInt32();
            var dims = new Dim[rank];
            for (int j = 0; j < rank; j++)
            {
                dims[j] = Dim.Of(r.ReadInt64());
            }

            long offset = r.ReadInt64();
            long length = r.ReadInt64();
            if (offset < 0 || length < 0 || offset + length > fileLength)
            {
                throw new InvalidDataException($"Weight {name} lies outside the package.");
            }

            var shape = new Shape(dims);
            if (type.ByteSize(shape.Elements()) != length)
            {
                throw new InvalidDataException($"Weight {name}: {length} bytes do not match {type}{shape}.");
            }

            weights.Add(new PackageWeightInfo(name, type, shape, offset, length));
        }
        return weights;
    }
}
