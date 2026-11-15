using System.Buffers.Binary;
using System.Numerics;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>One VPTQ linear module: its settings from <c>config_for_layers</c>, its tensors and the row decoder (see <see cref="HfVptqWeights"/>).</summary>
internal sealed unsafe class HfVptqModule
{
    private readonly string _name;
    private readonly DataType _type;
    private readonly int _rows;
    private readonly int _columns;
    private readonly int _vector;
    private readonly int _codebooks;
    private readonly int _groupSize;
    private readonly int _indexBits;
    private readonly int _residualBits;
    private readonly bool _packed;
    private readonly long _packedWords;
    private readonly long _indexRows;
    private readonly int _outlierSize;
    private readonly int _outlierVector;
    private readonly SafetensorsTensor _indices;
    private readonly SafetensorsTensor _centroids;
    private readonly SafetensorsTensor? _residualIndices;
    private readonly SafetensorsTensor? _residualCentroids;
    private readonly SafetensorsTensor? _outlierIndices;
    private readonly SafetensorsTensor? _outlierCentroids;
    private readonly int[]? _permutation;
    private readonly SafetensorsTensor? _scale;
    private readonly SafetensorsTensor? _bias;

    /// <exception cref="NotSupportedException">The settings or tensor layout are not readable.</exception>
    public HfVptqModule(string module, JsonElement layer, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        _name = module + ".weight";
        _centroids = tensors[module + ".centroids.weight"];
        _indices = tensors[module + ".indices"];
        _type = _centroids.DType switch
        {
            "F16" => DataType.F16,
            "BF16" => DataType.BF16,
            _ => throw new NotSupportedException($"{_name}: VPTQ centroids of dtype {_centroids.DType} are not supported (F16 or BF16)."),
        };
        _rows = HfPackedBits.Int(layer, "out_features", 0);
        _columns = HfPackedBits.Int(layer, "in_features", 0);
        _codebooks = HfPackedBits.Int(layer, "group_num", 1);
        _groupSize = HfPackedBits.Int(layer, "group_size", 0);
        _outlierSize = HfPackedBits.Int(layer, "outlier_size", 0);
        int[] vectors = Pair(layer, "vector_lens");
        int[] centroids = Pair(layer, "num_centroids");
        int[] residuals = Pair(layer, "num_res_centroids");
        _vector = vectors[1];
        _outlierVector = vectors[0];
        _packed = layer.TryGetProperty("is_indice_packed", out var packed) && packed.ValueKind == JsonValueKind.True;
        bool outliers = _outlierVector > 1 && centroids[0] > 0;
        bool residual = residuals[1] > 0;
        if (_rows <= 0 || _columns <= 0 || _vector <= 0 || _groupSize <= 0 || centroids[1] <= 0
            || (!outliers && _outlierSize != 0) || _outlierSize + _codebooks * _groupSize != _columns
            || (layer.TryGetProperty("vector_quant_dim", out var dim) && dim.GetString() != "out"))
        {
            throw new NotSupportedException($"{_name}: VPTQ settings {layer.GetRawText()} are not a supported layout.");
        }

        _indexRows = (_rows + _vector - 1) / _vector;
        _indexBits = BitOperations.Log2((uint)centroids[1]) + (BitOperations.IsPow2(centroids[1]) ? 0 : 1);
        _residualBits = residual ? BitOperations.Log2((uint)residuals[1]) + (BitOperations.IsPow2(residuals[1]) ? 0 : 1) : 0;
        if (_packed)
        {
            _packedWords = ((long)_groupSize * (_indexBits + _residualBits) + 31) / 32;
            Expect(_indices, "I32", _codebooks, _indexRows, _packedWords);
        }
        else
        {
            // indices_as_float stores the same 16-bit patterns typed as F16.
            Expect(_indices, _indices.DType == "F16" ? "F16" : "I16", _codebooks, _indexRows, _groupSize);
        }

        Expect(_centroids, _centroids.DType, _codebooks, (long)centroids[1] * _vector);
        if (residual)
        {
            _residualCentroids = Tensor(tensors, module + ".res_centroids.weight");
            Expect(_residualCentroids, _centroids.DType, _codebooks, (long)residuals[1] * _vector);
            if (!_packed)
            {
                _residualIndices = Tensor(tensors, module + ".res_indices");
                Expect(_residualIndices, _residualIndices.DType == "F16" ? "F16" : "I16", _codebooks, _indexRows, _groupSize);
            }
        }

        if (outliers)
        {
            _outlierCentroids = Tensor(tensors, module + ".outlier_centroids.weight");
            _outlierIndices = Tensor(tensors, module + ".outlier_indices");
            Expect(_outlierCentroids, _centroids.DType, 1, (long)centroids[0] * _outlierVector);
            Expect(_outlierIndices, _outlierIndices.DType == "F16" ? "F16" : "I16", 1, (_rows + _outlierVector - 1) / _outlierVector, _outlierSize);
        }

        if (layer.TryGetProperty("enable_perm", out var perm) && perm.ValueKind == JsonValueKind.True)
        {
            var stored = Tensor(tensors, module + ".perm");
            if (stored.Shape.Length != 1 || stored.Shape[0] != _columns || stored.DType is not ("I16" or "I64"))
            {
                throw new NotSupportedException($"{_name}: VPTQ permutation of dtype {stored.DType} and shape [{string.Join(", ", stored.Shape)}] is not supported.");
            }

            _permutation = new int[_columns];
            var seen = new bool[_columns];
            for (int p = 0; p < _columns; p++)
            {
                long target = stored.DType == "I16" ? BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(stored.Pointer + p * 2L, 2))
                    : BinaryPrimitives.ReadInt64LittleEndian(new ReadOnlySpan<byte>(stored.Pointer + p * 8L, 8));
                if (target < 0 || target >= _columns || seen[target])
                {
                    throw new NotSupportedException($"{_name}: VPTQ permutation is not a permutation of the columns.");
                }

                seen[target] = true;
                _permutation[p] = (int)target;
            }
        }

        if (layer.TryGetProperty("enable_norm", out var norm) && norm.ValueKind == JsonValueKind.True)
        {
            _scale = Tensor(tensors, module + ".weight_scale");
            _bias = Tensor(tensors, module + ".weight_bias");
            Expect(_scale, _centroids.DType, _columns);
            Expect(_bias, _centroids.DType, _columns);
        }
    }

    /// <summary>The dense weight [out features, in features] in the centroids' dtype, decoded on demand.</summary>
    public SafetensorsTensor ToTensor(SafetensorsTensor anchor)
    {
        var converted = new WeightInfo
        {
            Name = _name,
            Type = _type,
            Shape = new Shape([(Dim)_rows, (Dim)_columns]),
            Source = new HfRowConvertedSource(_rows, _type.ByteSize(_columns), DecodeRow),
        };
        return new SafetensorsTensor
        {
            Name = _name,
            DType = _centroids.DType,
            Type = _type,
            Shape = [_rows, _columns],
            Offset = anchor.Offset,
            ByteLength = converted.ByteSize,
            File = anchor.File,
            Converted = converted,
        };
    }

    private void DecodeRow(long row, Span<byte> destination)
    {
        var values = new float[_columns];
        long indexRow = row / _vector;
        int element = (int)(row % _vector);
        if (_outlierIndices is not null && _outlierCentroids is not null)
        {
            long outlierRow = row / _outlierVector;
            int outlierElement = (int)(row % _outlierVector);
            for (int c = 0; c < _outlierSize; c++)
            {
                int index = BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(_outlierIndices.Pointer + (outlierRow * _outlierSize + c) * 2, 2));
                values[c] = Read(_outlierCentroids, (long)index * _outlierVector + outlierElement);
            }
        }

        long centroidStride = _centroids.Shape[1];
        long residualStride = _residualCentroids?.Shape[1] ?? 0;
        for (int codebook = 0; codebook < _codebooks; codebook++)
        {
            long rowStart = (long)codebook * _indexRows + indexRow;
            for (int g = 0; g < _groupSize; g++)
            {
                var (index, residual) = Index(rowStart, g);
                float value = Read(_centroids, codebook * centroidStride + (long)index * _vector + element);
                if (_residualCentroids is not null)
                {
                    value = HfVptqWeights.Round(_type, value + Read(_residualCentroids, codebook * residualStride + (long)residual * _vector + element));
                }

                values[_outlierSize + codebook * _groupSize + g] = value;
            }
        }

        if (_permutation is not null)
        {
            var permuted = new float[_columns];
            for (int p = 0; p < _columns; p++)
            {
                permuted[_permutation[p]] = values[p];
            }

            values = permuted;
        }

        for (int j = 0; j < _columns; j++)
        {
            float value = values[j];
            if (_scale is not null && _bias is not null)
            {
                value = HfVptqWeights.Round(_type, HfVptqWeights.Round(_type, value * Read(_scale, j)) + Read(_bias, j));
            }

            if (_type == DataType.F16)
            {
                BinaryPrimitives.WriteHalfLittleEndian(destination[(j * 2)..], (Half)value);
            }
            else
            {
                BinaryPrimitives.WriteUInt16LittleEndian(destination[(j * 2)..], (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16));
            }
        }
    }

    private (int Index, int Residual) Index(long rowStart, int g)
    {
        if (!_packed)
        {
            int index = BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(_indices.Pointer + (rowStart * _groupSize + g) * 2, 2));
            int residual = _residualIndices is null ? 0
                : BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(_residualIndices.Pointer + (rowStart * _groupSize + g) * 2, 2));
            return (index, residual);
        }

        int total = _indexBits + _residualBits;
        long bit = (long)g * total;
        long word = bit / 32;
        byte* words = _indices.Pointer + rowStart * _packedWords * 4;
        ulong low = BinaryPrimitives.ReadUInt32LittleEndian(new ReadOnlySpan<byte>(words + word * 4, 4));
        ulong high = word + 1 < _packedWords ? BinaryPrimitives.ReadUInt32LittleEndian(new ReadOnlySpan<byte>(words + (word + 1) * 4, 4)) : 0;
        ulong value = ((high << 32) | low) >> (int)(bit % 32);
        int packedIndex = (int)(value & ((1UL << _indexBits) - 1));
        int packedResidual = _residualBits == 0 ? 0 : (int)((value >> _indexBits) & ((1UL << _residualBits) - 1));
        return (packedIndex, packedResidual);
    }

    private float Read(SafetensorsTensor tensor, long index)
    {
        ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(tensor.Pointer + index * 2, 2));
        return _type == DataType.F16 ? (float)BitConverter.UInt16BitsToHalf(bits) : BitConverter.Int32BitsToSingle(bits << 16);
    }

    private SafetensorsTensor Tensor(IReadOnlyDictionary<string, SafetensorsTensor> tensors, string name)
    {
        return tensors.TryGetValue(name, out var tensor) ? tensor
            : throw new NotSupportedException($"{_name}: VPTQ module without {name}.");
    }

    private void Expect(SafetensorsTensor tensor, string dtype, params long[] shape)
    {
        if (tensor.DType != dtype || !tensor.Shape.SequenceEqual(shape))
        {
            throw new NotSupportedException($"{_name}: VPTQ tensor {tensor.Name} is {tensor.DType} [{string.Join(", ", tensor.Shape)}], expected {dtype} [{string.Join(", ", shape)}].");
        }
    }

    private int[] Pair(JsonElement layer, string name)
    {
        if (!layer.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Array || value.GetArrayLength() != 2)
        {
            throw new NotSupportedException($"{_name}: VPTQ setting {name} is missing or not a pair.");
        }

        return [value[0].GetInt32(), value[1].GetInt32()];
    }
}
