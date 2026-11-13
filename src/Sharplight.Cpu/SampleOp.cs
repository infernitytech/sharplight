using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Token sampling per logits row (<see cref="StandardOps.Sample"/>), see <see cref="SamplingReference"/>.</summary>
internal sealed unsafe class SampleOp : CpuOp
{
    private readonly TRef _logits;
    private readonly TRef _params;
    private readonly TRef _tokens;
    private readonly TRef _logprobs;
    private readonly Action<int> _body;
    private int _vocab;

    public override string Name => "Sample";

    public SampleOp(TRef logits, TRef parameters, TRef tokens, TRef logprobs)
    {
        _logits = logits;
        _params = parameters;
        _tokens = tokens;
        _logprobs = logprobs;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _vocab = (int)_logits.Dim(1, p);
        CpuParallel.For((int)_logits.Dim(0, p), _body);
    }

    private void Body(int row)
    {
        var encoded = new ReadOnlySpan<int>((int*)_params.P + row * SamplingRow.Width, SamplingRow.Width);
        int* tokens = (int*)_tokens.P;
        float* logprobs = (float*)_logprobs.P;
        if (!SamplingRow.TryRead(encoded, out var sampling))
        {
            tokens[row] = -1;
            logprobs[row] = 0;
            return;
        }

        var x = new ReadOnlySpan<float>((float*)_logits.P + (long)row * _vocab, _vocab);
        tokens[row] = SamplingReference.Sample(x, sampling, out logprobs[row]);
    }
}
