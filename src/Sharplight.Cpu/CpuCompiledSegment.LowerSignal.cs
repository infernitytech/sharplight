using Sharplight.Graph;

namespace Sharplight.Cpu;

// Signal processing.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerDft(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new DftOp(x, y, (int)a.Get<long>("length"), (int)a.Get<long>("out_bins"), a.Get("inverse", false), a.Get("real_output", false)), x, y);
        _refs[n.Output] = y;
    }

    private void LowerConv1d(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var w = Dense(RefF32(n.Inputs[1]));
        TRef? bias = n.Inputs.Length > 2 ? Dense(RefF32(n.Inputs[2])) : null;
        var y = NewDense(n.Output);
        AddStep(new Conv1dOp(x, w, bias, y, (int)a.Get("stride", 1L), (int)a.Get("pad_left", 0L), (int)a.Get("dilation", 1L), (int)a.Get("groups", 1L), a.Get("pad_mode", "zeros") == "reflect"),
            Touched(x, y, w, bias));
        _refs[n.Output] = y;
    }

    private void LowerSequenceConv2d(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var w = Dense(RefF32(n.Inputs[1]));
        TRef? bias = n.Inputs.Length > 2 ? Dense(RefF32(n.Inputs[2])) : null;
        var y = NewDense(n.Output);
        var strides = a.Get("strides", new long[] { 1, 1 });
        var pads = a.Get("pads", new long[] { 0, 0, 0, 0 });
        AddStep(new SequenceConv2dOp(x, w, bias, y, (int)strides[0], (int)strides[1], (int)pads[0], (int)pads[1], (int)a.Get("groups", 1L)),
            Touched(x, y, w, bias));
        _refs[n.Output] = y;
    }

    private void LowerRelativeAttention(Node n)
    {
        var inputs = n.Inputs.Take(5).Select(i => Dense(RefF32(i))).ToArray();
        var spans = n.Inputs.Length > 5 ? Dense(Ref(n.Inputs[5])) : null;
        var y = NewDense(n.Output);
        AddStep(new RelativeAttentionOp(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], spans, y, n.Attributes.Get("scale", 1f)),
            Touched(inputs[0], y, inputs[1], inputs[2], inputs[3], inputs[4], spans));
        _refs[n.Output] = y;
    }

    private void LowerConvTranspose1d(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var w = Dense(RefF32(n.Inputs[1]));
        TRef? bias = n.Inputs.Length > 2 ? Dense(RefF32(n.Inputs[2])) : null;
        var y = NewDense(n.Output);
        AddStep(new ConvTranspose1dOp(x, w, bias, y, (int)a.Get("stride", 1L), (int)a.Get("crop_left", 0L), (int)a.Get("groups", 1L)), Touched(x, y, w, bias));
        _refs[n.Output] = y;
    }

    private void LowerInterpolate1d(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new Interpolate1dOp(x, y, a.Get("mode", "nearest") == "linear", (int)a.Get("up", 1L), (int)a.Get("down", 1L)), x, y);
        _refs[n.Output] = y;
    }

    private void LowerLstm(Node n)
    {
        var gates = Dense(RefF32(n.Inputs[0]));
        var w = Dense(RefF32(n.Inputs[1]));
        var y = NewDense(n.Output);
        AddStep(new LstmOp(gates, w, y, n.Attributes.Get("reverse", false)), gates, y, w);
        _refs[n.Output] = y;
    }
}
