using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Batched linear attention recurrence (see <c>StandardOps.LinearAttention</c>): one task per (batch, key/value head) walks the tokens,
/// updating its [Dk, Dv] state and producing the outputs of the query heads that read it.
/// </summary>
internal sealed unsafe class LinearAttentionOp : CpuOp
{
    private readonly TRef _q;
    private readonly TRef _k;
    private readonly TRef _v;
    private readonly TRef? _state;
    private readonly TRef? _decay;
    private readonly TRef? _beta;
    private readonly TRef _y;
    private readonly TRef _stateOut;
    private readonly int _qHeads;
    private readonly int _kvHeads;
    private readonly float _scale;
    private readonly Action<int> _body;
    private long _tokens;
    public override string Name => "LinearAttention";

    public LinearAttentionOp(TRef q, TRef k, TRef v, TRef? state, TRef? decay, TRef? beta, TRef y, TRef stateOut, int qHeads, int kvHeads, float scale)
    {
        _q = q;
        _k = k;
        _v = v;
        _state = state;
        _decay = decay;
        _beta = beta;
        _y = y;
        _stateOut = stateOut;
        _qHeads = qHeads;
        _kvHeads = kvHeads;
        _scale = scale;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_q, p);
        _tokens = d[1];
        CpuParallel.For((int)(d[0] * _kvHeads), _body);
    }

    private void Body(int task)
    {
        int b = task / _kvHeads;
        int h = task % _kvHeads;
        int dk = (int)(_k.Dims[2] / _kvHeads);
        int dv = (int)(_v.Dims[2] / _kvHeads);
        int group = _qHeads / _kvHeads;
        long qRow = _q.Dims[2];
        long kRow = _k.Dims[2];
        long vRow = _v.Dims[2];
        float* s = (float*)_stateOut.P + (long)task * dk * dv;
        if (_state is not null)
        {
            Buffer.MemoryCopy((float*)_state.P + (long)task * dk * dv, s, (long)dk * dv * 4, (long)dk * dv * 4);
        }
        else
        {
            new Span<float>(s, dk * dv).Clear();
        }

        var update = new float[dv];
        for (long t = 0; t < _tokens; t++)
        {
            long token = b * _tokens + t;
            float* k = (float*)_k.P + token * kRow + h * dk;
            float* v = (float*)_v.P + token * vRow + h * dv;
            if (_decay is not null)
            {
                long width = _decay.Dims[2];
                float* g = (float*)_decay.P + token * width;
                bool perDim = width == (long)_kvHeads * dk;
                for (int i = 0; i < dk; i++)
                {
                    float factor = MathF.Exp(perDim ? g[h * dk + i] : g[h]);
                    for (int m = 0; m < dv; m++)
                    {
                        s[i * dv + m] *= factor;
                    }
                }
            }

            if (_beta is not null)
            {
                long width = _beta.Dims[2];
                float beta = ((float*)_beta.P)[token * width + (width == 1 ? 0 : h)];
                for (int m = 0; m < dv; m++)
                {
                    float retrieved = 0;
                    for (int i = 0; i < dk; i++)
                    {
                        retrieved += s[i * dv + m] * k[i];
                    }

                    update[m] = beta * (v[m] - retrieved);
                }
            }
            else
            {
                new Span<float>(v, dv).CopyTo(update);
            }

            for (int i = 0; i < dk; i++)
            {
                for (int m = 0; m < dv; m++)
                {
                    s[i * dv + m] += k[i] * update[m];
                }
            }

            for (int r = 0; r < group; r++)
            {
                int head = h * group + r;
                float* q = (float*)_q.P + token * qRow + head * dk;
                float* y = (float*)_y.P + token * _qHeads * dv + head * dv;
                for (int m = 0; m < dv; m++)
                {
                    float acc = 0;
                    for (int i = 0; i < dk; i++)
                    {
                        acc += q[i] * s[i * dv + m];
                    }

                    y[m] = _scale * acc;
                }
            }
        }
    }
}
