namespace Sharplight.Onnx;

// Tensor operators without a better home: Det.
internal sealed partial class OnnxTranslator
{
    /// <summary>Det: determinants of [..., M, M] (the IR Det op; CPU).</summary>
    private void Det(OnnxNode n, TranslatedTensor?[] ins)
    {
        SetDyn(n, 0, Emit("Det", [X(ins, 0)], null, Nm(n)));
    }
}
