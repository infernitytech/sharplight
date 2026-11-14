using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

// The batched LinearAttention recurrence (linear, gated, delta and gated delta rules) with its state passed in and out.
public sealed partial class GraphLowerer
{
    /// <summary>
    /// LinearAttention (see <c>StandardOps.LinearAttention</c>). Every column m of a key/value head's [Dk, Dv] state evolves on its own:
    /// the decay scales the column, the delta rule's retrieval Sᵀk and the update k ⊗ v' touch only column m, and output column m of
    /// every query head reading that state is Sᵀq over it. So one thread per (batch, key/value head, value column) walks the tokens,
    /// keeping its column in the state output.
    /// </summary>
    private void LowerLinearAttention(Node n)
    {
        var a = n.Attributes;
        var q = AsDenseF32(Ref(n.Inputs[0]), "linear-attention-q");
        var k = AsDenseF32(Ref(n.Inputs[1]), "linear-attention-k");
        var v = AsDenseF32(Ref(n.Inputs[2]), "linear-attention-v");
        var ins = new List<TensorRef> { q, k, v };
        int next = 3;
        (string Name, long Width)? Optional(string flag)
        {
            if (!a.Get(flag, false))
            {
                return null;
            }

            var r = AsDenseF32(Ref(n.Inputs[next++]), "linear-attention-" + flag);
            ins.Add(r);
            return ($"p.in{ins.Count - 1}", r.Rank == 3 ? r.Shape[2] : 0);
        }

        string? state = Optional("state")?.Name;
        var decayInput = Optional("gated");
        var betaInput = Optional("delta");
        string? decay = decayInput?.Name;
        string? beta = betaInput?.Name;
        long decayWidth = decayInput?.Width ?? 0;
        long betaWidth = betaInput?.Width ?? 0;
        var y = Produce(n.Outputs[0]);
        var stateOut = Produce(n.Outputs[1]);
        long qHeads = a.Get<long>("q_heads");
        long kvHeads = a.Get<long>("kv_heads");
        long dk = k.Shape[2] / kvHeads;
        long dv = v.Shape[2] / kvHeads;
        long tokens = q.Shape[1];
        long group = qHeads / kvHeads;
        bool perDimDecay = decayWidth == kvHeads * dk;
        string decayFactor = perDimDecay ? $"exp({decay}[tok * {decayWidth}u + h * {dk}u + i])" : "f";
        string body = $$"""
              uint b = d0, h = d1, m = d2;
              uint sb = (b * {{kvHeads}}u + h) * {{dk * dv}}u + m;
              for (uint i = 0; i < {{dk}}u; i++) { p.out1[sb + i * {{dv}}u] = {{(state is null ? "0.0" : $"{state}[sb + i * {dv}u]")}}; }
              for (uint t = 0; t < {{tokens}}u; t++) {
                uint tok = b * {{tokens}}u + t;
                uint kb = tok * {{k.Shape[2]}}u + h * {{dk}}u;
                float u = p.in2[tok * {{v.Shape[2]}}u + h * {{dv}}u + m];
            {{(decay is not null && !perDimDecay ? $"    float f = exp({decay}[tok * {decayWidth}u + h]);" : "")}}
            {{(decay is not null || beta is not null ? $$"""
                float r = 0.0;
                for (uint i = 0; i < {{dk}}u; i++) {
                  float s = p.out1[sb + i * {{dv}}u]{{(decay is null ? "" : $" * {decayFactor}")}};
                  p.out1[sb + i * {{dv}}u] = s;
                  r += s * p.in1[kb + i];
                }
            """ : "")}}
            {{(beta is null ? "" : $"    u = {beta}[tok * {betaWidth}u + {(betaWidth == 1 ? "0u" : "h")}] * (u - r);")}}
                for (uint i = 0; i < {{dk}}u; i++) { p.out1[sb + i * {{dv}}u] += p.in1[kb + i] * u; }
                for (uint g = 0; g < {{group}}u; g++) {
                  uint head = h * {{group}}u + g;
                  uint qb = tok * {{q.Shape[2]}}u + head * {{dk}}u;
                  float acc = 0.0;
                  for (uint i = 0; i < {{dk}}u; i++) { acc += p.in0[qb + i] * p.out1[sb + i * {{dv}}u]; }
                  p.out0[tok * {{qHeads * dv}}u + head * {{dv}}u + m] = {{F(a.Get("scale", 1f))}} * acc;
                }
              }
            """;
        EmitGen("linear_attention", [n], [q.Shape[0], kvHeads, dv], q.Symbol, ins, [y, stateOut], body, outsAreReads: true);
    }
}
