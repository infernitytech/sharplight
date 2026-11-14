namespace Sharplight.Onnx;

// Constant generators of the signal domain: window functions and the mel filter bank (their inputs are sizes, so they fold at import).
internal static partial class OnnxFold
{
    private static OnnxConst FoldWindow(OnnxNode n, IReadOnlyList<OnnxConst?> ins)
    {
        long size = Input(n, ins, 0).ScalarInt();
        var w = SignalMath.Window(n.OpType, size, n.Int("periodic", 1) != 0);
        return OnnxConst.FromFloats(OnnxType.Float, [size], w).CastTo((int)n.Int("output_datatype", OnnxType.Float));
    }

    private static OnnxConst FoldMelWeightMatrix(OnnxNode n, IReadOnlyList<OnnxConst?> ins)
    {
        long melBins = Input(n, ins, 0).ScalarInt();
        long dftLength = Input(n, ins, 1).ScalarInt();
        var w = SignalMath.MelWeights(melBins, dftLength, Input(n, ins, 2).ScalarInt(), Input(n, ins, 3).Floats()[0], Input(n, ins, 4).Floats()[0]);
        return OnnxConst.FromFloats(OnnxType.Float, [(dftLength / 2) + 1, melBins], w).CastTo((int)n.Int("output_datatype", OnnxType.Float));
    }
}
