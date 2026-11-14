using Sharplight.Graph;

namespace Sharplight.Architectures;

internal sealed partial class LatentDecoderGraphBuilder
{
    /// <summary>Dense MLP from the weights below <paramref name="p"/> + <paramref name="block"/> (gated unless <c>mlp.gated</c> is false).</summary>
    private Value Mlp(GraphBuilder b, Value h, string p, string block, string name)
    {
        var up = b.Linear(h, p + block + "up.weight");
        var act = _hp.Gated ? _activation.ActivateGated(b, b.Linear(h, p + block + "gate.weight"), up, p) : _activation.Activate(b, up, p);
        return b.Linear(act, p + block + "down.weight", name: name);
    }

    /// <summary>
    /// Routed experts plus shared experts through the shared <see cref="MoeBlock"/> (its <see cref="MoeOptions"/> cover DeepSeek's
    /// softmax / sigmoid scores, selection bias, group-limited top-k, renormalization and scaling).
    /// </summary>
    /// <exception cref="InvalidDataException">MoE weights are present but the expert counts or groups are inconsistent.</exception>
    private Value Experts(GraphBuilder b, Value h, string p)
    {
        if (!_hp.Moe.IsValid)
        {
            throw new InvalidDataException("latent_decoder: MoE weights present but moe.expert_count / moe.expert_used_count / moe.group_* are missing or inconsistent.");
        }

        int layer = int.Parse(p.Split('.')[^2], System.Globalization.CultureInfo.InvariantCulture);
        bool clamped = !p.StartsWith("mtp.", StringComparison.Ordinal) && layer < _hp.MoeSwigluClamp.Length && _hp.MoeSwigluClamp[layer] > 0f;
        var experts = clamped ? new ClampedSwigluActivation(_hp.MoeSwigluClamp[layer], _hp.MoeSwigluClampInput) : _activation;
        return MoeBlock.Build(b, h, p, "mlp.", _hp.Moe, _hp.HiddenSize, experts, sharedActivation: _activation);
    }
}
