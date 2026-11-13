namespace Sharplight.Onnx.Conformance;

/// <summary>A tensor of an ONNX test data set (input_k.pb / output_k.pb), decoded to doubles.</summary>
/// <param name="ElemType">ONNX element type id.</param>
/// <param name="Dims">ONNX dims (empty for a scalar).</param>
/// <param name="Values">Element values in row-major order.</param>
internal sealed record TestTensor(int ElemType, long[] Dims, double[] Values)
{
    public long Elements => Dims.Aggregate(1L, (a, b) => a * b);

    public bool IsFloat => OnnxType.IsFloat(ElemType);

    /// <summary>Parses a serialized TensorProto; throws <see cref="NotSupportedException"/> for strings and non-tensors.</summary>
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
            throw new NotSupportedException($"{what}: string tensors are not supported by the harness");
        }

        if (tensor.ElemType == 0)
        {
            throw new NotSupportedException($"{what} is not a plain tensor (sequence, map or optional)");
        }

        var values = OnnxConst.FromTensor(tensor).Floats();
        return new TestTensor(tensor.ElemType, tensor.Dims, values);
    }
}
