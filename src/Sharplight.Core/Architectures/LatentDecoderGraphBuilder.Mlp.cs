using Sharplight.Graph;

namespace Sharplight.Architectures;

internal sealed partial class LatentDecoderGraphBuilder
{
    /// <summary>
    /// The feed-forward block of the layer below <paramref name="p"/>: routed experts when the layer has a router, the dense MLP otherwise; plus the
    /// layer's reasoning-effort steering vector <c>mlp.steering</c> [H] times <c>mlp.steering_scale</c> when it has one (Raon-1).
    /// </summary>
    private Value FeedForward(GraphBuilder b, Value h, string p)
    {
        var ffn = MoeBlock.IsPresent(b, p + "mlp.") ? Experts(b, h, p) : Mlp(b, h, p, "mlp.", $"{p}ffn_out");
        if (_steeringScale != 0f && b.TryWeight(p + "mlp.steering") is { } steering)
        {
            ffn = b.Add(ffn, b.Scale(steering, _steeringScale), $"{p}ffn_steered");
        }

        return ffn;
    }

    /// <summary>Dense MLP from the weights below <paramref name="p"/> + <paramref name="block"/> (gated unless <c>mlp.gated</c> is false).</summary>
    private Value Mlp(GraphBuilder b, Value h, string p, string block, string name)
    {
        var up = b.Linear(h, p + block + "up.weight");
        var activation = MlpActivation(p);
        var act = _hp.Gated ? activation.ActivateGated(b, b.Linear(h, p + block + "gate.weight"), up, p) : activation.Activate(b, up, p);
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

        int layer = TrunkLayer(p);
        bool clamped = layer >= 0 && layer < _hp.MoeSwigluClamp.Length && _hp.MoeSwigluClamp[layer] > 0f;
        var shared = MlpActivation(p);
        var experts = clamped ? new ClampedSwigluActivation(_hp.MoeSwigluClamp[layer], _hp.MoeSwigluClampInput) : shared;
        return MoeBlock.Build(b, h, p, "mlp.", _hp.Moe, _hp.HiddenSize, experts, sharedActivation: shared, tokens: layer >= 0 ? _tokens : null);
    }

    /// <summary>
    /// The activation of the dense MLP and shared experts of the layer below <paramref name="p"/>: SwiGLU with clamped operands when
    /// <c>mlp.swiglu_clamp</c> gives the layer a limit (with <c>mlp.swiglu_clamp_input</c> on the gate before the SiLU, see
    /// <see cref="ClampedSwigluActivation"/>), the model's activation otherwise.
    /// </summary>
    private IMlpActivation MlpActivation(string p)
    {
        int layer = TrunkLayer(p);
        return layer >= 0 && layer < _hp.MlpSwigluClamp.Length && _hp.MlpSwigluClamp[layer] > 0f
            ? new ClampedSwigluActivation(_hp.MlpSwigluClamp[layer], _hp.MlpSwigluClampInput)
            : _activation;
    }

    // The trunk layer index of a weight prefix ("layers.{i}."), -1 for the draft layers ("mtp.layers.{j}.").
    private static int TrunkLayer(string p)
        => p.StartsWith("mtp.", StringComparison.Ordinal) ? -1 : int.Parse(p.Split('.')[^2], System.Globalization.CultureInfo.InvariantCulture);
}
