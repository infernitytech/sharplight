using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the diffusion head of VibeVoice, which denoises one acoustic latent per frame given the language model's hidden state: noisy latents [rows, latent] and the sinusoidal embedding
/// of the timestep [1, frequencies] (computed on the host) go in; the predicted velocity [rows, latent] comes out. The latents and the condition are projected to the head's width, the
/// condition plus the timestep MLP's output modulates (adaptive RMS norm: shift, scale, gate) a stack of gated feed-forward layers, and a final modulated layer projects back.
/// </summary>
internal static class VibeVoiceDiffusionGraph
{
    /// <summary>The graph's name.</summary>
    public const string Graph = "diffusion.head";

    /// <summary>The noisy latents input.</summary>
    public const string NoisyInput = "noisy";

    /// <summary>The timestep embedding input.</summary>
    public const string TimestepInput = "timestep";

    /// <summary>The condition input (the language model's hidden states).</summary>
    public const string ConditionInput = "condition";

    /// <summary>The predicted velocity output.</summary>
    public const string Output = "velocity";

    /// <summary>Builds the head.</summary>
    /// <param name="weights">Finds a weight by name.</param>
    /// <param name="layers">The layers of the stack.</param>
    /// <param name="latent">The width of the latents.</param>
    /// <param name="hidden">The width of the condition and of the head.</param>
    /// <param name="frequencies">The width of the sinusoidal timestep embedding.</param>
    /// <param name="eps">The RMS norm epsilon.</param>
    /// <param name="rows">The rows per run (the conditional and the unconditional one).</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, int layers, int latent, int hidden, int frequencies, float eps, int rows)
    {
        var b = new GraphBuilder(Graph, weights);
        var noisy = b.Input(NoisyInput, InputRole.Other, DataType.F32, new Shape(rows, latent));
        var timestep = b.Input(TimestepInput, InputRole.Other, DataType.F32, new Shape(1, frequencies));
        var condition = b.Input(ConditionInput, InputRole.Other, DataType.F32, new Shape(rows, hidden));
        var x = b.Linear(noisy, "diffusion.noisy_proj.weight");
        var time = b.Linear(b.Silu(b.Linear(timestep, "diffusion.time.fc1.weight")), "diffusion.time.fc2.weight");
        var c = b.Add(b.Linear(condition, "diffusion.cond_proj.weight"), time);
        var modulation = b.Silu(c);
        for (int il = 0; il < layers; il++)
        {
            using var _ = b.Group($"diffusion.layers.{il}");
            string n = $"diffusion.layers.{il}.";
            var chunks = b.Linear(modulation, n + "ada.weight");
            var shift = b.Slice(chunks, 1, 0, hidden);
            var scale = b.Slice(chunks, 1, hidden, hidden);
            var gate = b.Slice(chunks, 1, 2L * hidden, hidden);
            var h = b.Add(b.Mul(b.RmsNorm(x, b.Weight(n + "norm.weight"), eps), b.Scale(scale, 1f, 1f)), shift);
            var ffn = b.Linear(b.Mul(b.Silu(b.Linear(h, n + "ffn.gate.weight")), b.Linear(h, n + "ffn.up.weight")), n + "ffn.down.weight");
            x = b.Add(x, b.Mul(gate, ffn));
        }

        var final = b.Linear(modulation, "diffusion.final.ada.weight");
        var finalShift = b.Slice(final, 1, 0, hidden);
        var finalScale = b.Slice(final, 1, hidden, hidden);
        var normed = b.Add(b.Mul(b.RmsNorm(x, null, eps), b.Scale(finalScale, 1f, 1f)), finalShift);
        b.Output(Output, b.Linear(normed, "diffusion.final.out.weight"));
        return b.Build();
    }
}
