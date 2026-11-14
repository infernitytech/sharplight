namespace Sharplight.Onnx.Conformance;

/// <summary>
/// A value of an ONNX test data set: a tensor, a sequence of tensors, or an empty optional (an optional holding a value is that value).
/// </summary>
/// <param name="Tensor">The tensor; null for sequences and empty optionals.</param>
/// <param name="Elements">The elements of a sequence; null for tensors and empty optionals.</param>
internal sealed record TestValue(TestTensor? Tensor, IReadOnlyList<TestTensor>? Elements)
{
    // DataType enum of SequenceProto / OptionalProto (onnx-data.proto).
    private const int TensorElement = 1;
    private const int SequenceElement = 3;

    public bool IsNone => Tensor is null && Elements is null;

    /// <summary>Reads a TensorProto, SequenceProto or OptionalProto file, as stated by the graph's value info.</summary>
    /// <exception cref="NotSupportedException">Maps, sparse tensors, nested sequences or strings.</exception>
    public static TestValue Load(string path, OnnxValueInfo info)
    {
        var bytes = File.ReadAllBytes(path);
        return info.Kind switch
        {
            OnnxValueKind.Sequence => new TestValue(null, ReadSequence(bytes)),
            OnnxValueKind.Optional => ReadOptional(bytes),
            OnnxValueKind.Tensor or OnnxValueKind.Unknown => new TestValue(TestTensor.Parse(bytes, Path.GetFileName(path)), null),
            _ => throw new NotSupportedException($"{Path.GetFileName(path)} is a {info.Kind.ToString().ToLowerInvariant()} value"),
        };
    }

    /// <summary>SequenceProto: name = 1, elem_type = 2, tensor_values = 3, sparse_tensor_values = 4, sequence_values = 5, map_values = 6, optional_values = 7.</summary>
    private static List<TestTensor> ReadSequence(byte[] bytes)
    {
        var elements = new List<TestTensor>();
        var reader = new ProtoReader(bytes);
        while (reader.Next(out int field, out var wire))
        {
            if (field == 3 && wire == WireType.LengthDelimited)
            {
                elements.Add(TestTensor.Parse(reader.ReadBytes(out _).ToArray(), $"sequence element {elements.Count}"));
            }
            else if (field is 4 or 5 or 6 or 7)
            {
                throw new NotSupportedException("a sequence of sparse tensors, sequences, maps or optionals");
            }
            else
            {
                reader.Skip(wire);
            }
        }

        return elements;
    }

    /// <summary>OptionalProto: name = 1, elem_type = 2, tensor_value = 3, sparse_tensor_value = 4, sequence_value = 5, map_value = 6, optional_value = 7.</summary>
    private static TestValue ReadOptional(byte[] bytes)
    {
        var reader = new ProtoReader(bytes);
        long elementType = 0;
        var value = new TestValue(null, null);
        while (reader.Next(out int field, out var wire))
        {
            if (field == 2 && wire == WireType.Varint)
            {
                elementType = reader.ReadInt64();
            }
            else if (field == 3 && wire == WireType.LengthDelimited)
            {
                value = new TestValue(TestTensor.Parse(reader.ReadBytes(out _).ToArray(), "optional tensor"), null);
            }
            else if (field == 5 && wire == WireType.LengthDelimited)
            {
                value = new TestValue(null, ReadSequence(reader.ReadBytes(out _).ToArray()));
            }
            else if (field is 4 or 6 or 7)
            {
                throw new NotSupportedException("an optional sparse tensor, map or optional");
            }
            else
            {
                reader.Skip(wire);
            }
        }

        return elementType is 0 or TensorElement or SequenceElement ? value : throw new NotSupportedException($"an optional of element type {elementType}");
    }
}
