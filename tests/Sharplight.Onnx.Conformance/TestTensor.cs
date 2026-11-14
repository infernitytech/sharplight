namespace Sharplight.Onnx.Conformance;

/// <summary>A tensor of an ONNX test data set (input_k.pb / output_k.pb), decoded to doubles or strings.</summary>
/// <param name="ElemType">ONNX element type id.</param>
/// <param name="Dims">ONNX dims (empty for a scalar).</param>
/// <param name="Values">Element values in row-major order (empty for a string tensor).</param>
/// <param name="Strings">Elements of a string tensor in row-major order; null for other types.</param>
internal sealed record TestTensor(int ElemType, long[] Dims, double[] Values, string[]? Strings = null)
{
    public long Elements => Dims.Aggregate(1L, (a, b) => a * b);

    public bool IsFloat => OnnxType.IsFloat(ElemType);

    /// <summary>Parses a serialized TensorProto; throws <see cref="NotSupportedException"/> for non-tensors.</summary>
    /// <param name="bytes">The TensorProto.</param>
    /// <param name="what">The file or element it was read from (for error messages).</param>
    public static TestTensor Parse(byte[] bytes, string what)
    {
        OnnxTensor tensor;
        try
        {
            tensor = OnnxParser.ParseTensorProto(bytes);
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException)
        {
            throw new NotSupportedException($"{what} is not a plain tensor (sequence, map or optional?): {e.Message}", e);
        }

        if (tensor.ElemType == OnnxType.String)
        {
            return new TestTensor(tensor.ElemType, tensor.Dims, [], tensor.Strings);
        }

        if (tensor.ElemType == 0)
        {
            throw new NotSupportedException($"{what} is not a plain tensor (sequence, map or optional)");
        }

        var values = OnnxConst.FromTensor(tensor).Floats();
        return new TestTensor(tensor.ElemType, tensor.Dims, values);
    }
}
