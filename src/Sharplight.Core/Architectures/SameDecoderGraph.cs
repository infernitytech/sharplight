using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the decoder of a SAME autoencoder (Stable Audio 3) over a fixed window of latent frames: the latents scaled by the bottleneck's scale and projected to the
/// transformer width, each frame followed by <see cref="SameDecoderParameters.Stride"/> new tokens (the learned token plus noise), transformer blocks (DynamicTanh norms,
/// differential attention with normed queries and keys and a partial rotary embedding, a SwiGLU or sine-gated feed-forward layer) over chunks of
/// <c>ChunkSize / Stride</c> frames (the second half of the blocks on chunks shifted by half a chunk, the edge frames repeated) or over a sliding window, then the new
/// tokens through the output convolution to patches of the waveform.
/// </summary>
/// <remarks>
/// Weights under the prefix: <c>latent_scale</c>, <c>decoder.in.{weight, bias}</c>, <c>decoder.new_tokens</c> <c>[n, width]</c>, <c>decoder.layers.{i}.{attention_norm,
/// mlp_norm, attention.q_norm, attention.k_norm}.{alpha, weight, bias}</c> (DynamicTanh) or <c>.weight</c> (RMS norms), <c>decoder.layers.{i}.attention.{qkv, o}.weight</c>
/// (the fused projection holds q, k, v and, for differential attention, the second q and k), <c>decoder.layers.{i}.mlp.{up, down}.{weight, bias}</c> (the up
/// projection's value half before its gate half) and <c>decoder.out.{weight, bias}</c> (the output convolution <c>[patch rows, width, kernel]</c>).
/// </remarks>
internal static class SameDecoderGraph
{
    /// <summary>
    /// Builds the decoder: latents f32 <c>[frames, latent channels]</c> and the new tokens' noise f32 <c>[frames · stride, width]</c> (standard normal; scaled by the
    /// model's mask noise, zeros for a deterministic decode) to audio f32 <c>[frames · stride · patch, channels]</c>.
    /// </summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="latent">The latents.</param>
    /// <param name="noise">The new tokens' noise.</param>
    /// <param name="p">The decoder's hyper-parameters.</param>
    /// <param name="prefix">The weights' prefix (e.g. <c>vae.</c>).</param>
    /// <param name="frames">The fixed number of latent frames (a multiple of the frames per chunk).</param>
    /// <param name="audioChannels">The audio channels.</param>
    public static Value Build(GraphBuilder b, Value latent, Value noise, SameDecoderParameters p, string prefix, int frames, int audioChannels)
    {
        string d = prefix + "decoder.";
        int width = p.Width;
        int sub = p.SubChunk;
        long rows = (long)frames * sub;
        Value x;
        using (b.Group(d + "input"))
        {
            x = b.Mul(latent, b.Weight(prefix + "latent_scale"));
            x = b.Linear(x, d + "in.weight", d + "in.bias");
            var newTokens = b.Weight(d + "new_tokens");
            long tokens = newTokens.Shape[0].Fixed;
            var noisy = b.Add(b.Reshape(newTokens, 1, tokens, width), b.Scale(b.Reshape(noise, frames, p.Stride, width), p.MaskNoise));
            x = b.Reshape(b.Concat(1, b.Reshape(x, frames, 1, width), noisy), rows, width);
        }

        if (p.SlidingWindow.Length > 0)
        {
            // Sliding-window attention over the whole window: a token sees the tokens of the frames within the window on either side.
            long reach = Math.Max(p.SlidingWindow[0], p.SlidingWindow[^1]) * sub;
            var positions = Positions(b, "positions", rows, rows);
            var window = Attributes.From(("scale", 1f / MathF.Sqrt(p.HeadDim)), ("causal", false), ("window", reach + 1));
            for (int il = 0; il < p.Layers; il++)
            {
                x = Block(b, x, p, d, il, positions, window);
            }
        }
        else
        {
            long chunk = (long)(p.ChunkSize / p.Stride) * sub;
            var chunked = Attributes.From(("scale", 1f / MathF.Sqrt(p.HeadDim)), ("causal", false), ("sequence_rows", chunk), ("sequence_keys", chunk));
            int split = p.MidpointShift ? p.Layers / 2 : p.Layers;
            var positions = Positions(b, "chunk_positions", rows, chunk);
            for (int il = 0; il < split; il++)
            {
                x = Block(b, x, p, d, il, positions, chunked);
            }

            if (split < p.Layers)
            {
                // Chunks shifted by half a chunk: the first and the last sub-chunk repeated at the edges, cut off again afterwards.
                long shift = chunk / 2;
                x = b.Concat(0, b.Slice(x, 0, 0, shift), x, b.Slice(x, 0, rows - shift, shift));
                var shifted = Positions(b, "shifted_positions", rows + (2 * shift), chunk);
                for (int il = split; il < p.Layers; il++)
                {
                    x = Block(b, x, p, d, il, shifted, chunked);
                }

                x = b.Slice(x, 0, shift, rows);
            }
        }

        using (b.Group(d + "output"))
        {
            // Keep the new tokens of every sub-chunk, map them to patch rows and unfold the patches: [rows, (c h)] → [rows · h, c].
            x = b.Contiguous(b.Slice(b.Reshape(x, frames, sub, width), 1, 1, p.Stride));
            x = b.Reshape(x, (long)frames * p.Stride, width);
            long pad = (p.MappingKernel - 1) / 2;
            x = b.Emit1("Conv1d", [x, b.Weight(d + "out.weight"), b.Weight(d + "out.bias")], Attributes.From(("pad_left", pad), ("pad_right", pad)));
            int patch = p.PatchSize;
            x = b.Contiguous(b.Permute(b.Reshape(x, (long)frames * p.Stride, audioChannels, patch), 0, 2, 1));
            x = b.Reshape(x, (long)frames * p.Stride * patch, audioChannels);
            return p.SoftClip ? b.Unary("Tanh", x) : x;
        }
    }

    private static Value Block(GraphBuilder b, Value x, SameDecoderParameters p, string d, int il, Value positions, Attributes attention)
    {
        using var _ = b.Group($"{d}layers.{il}");
        string n = $"{d}layers.{il}.";
        int heads = p.Width / p.HeadDim;
        var rope = Attributes.From(("dims", (long)(p.HeadDim / 2)), ("base", SameDecoderParameters.RopeBase), ("style", "neox"));
        var h = Norm(b, x, p, n + "attention_norm");
        var qkv = b.MatMul(h, b.Weight(n + "attention.qkv.weight"));
        Value Part(int i) => b.Reshape(b.Slice(qkv, 1, (long)i * p.Width, p.Width), 0, heads, p.HeadDim);
        Value Query(int i) => b.Emit1("Rope", [Norm(b, Part(i), p, n + "attention.q_norm"), positions], rope);
        Value Key(int i) => b.Emit1("Rope", [Norm(b, Part(i), p, n + "attention.k_norm"), positions], rope);
        var v = Part(2);
        var attended = b.Emit1("Attention", [Query(0), Key(1), v], attention);
        if (p.Differential)
        {
            attended = b.Sub(attended, b.Emit1("Attention", [Query(3), Key(4), v], attention));
        }

        x = b.Add(x, b.MatMul(b.Reshape(attended, 0, p.Width), b.Weight(n + "attention.o.weight")));

        h = Norm(b, x, p, n + "mlp_norm");
        var up = b.Linear(h, n + "mlp.up.weight", n + "mlp.up.bias");
        long inner = b.Weight(n + "mlp.down.weight").Shape[1].Fixed;
        bool sinusoidal = p.Layers - il < p.SinusoidalBlocks;
        var gate = b.Slice(up, 1, inner, inner);
        var gated = b.Mul(b.Slice(up, 1, 0, inner), sinusoidal ? b.Unary("Sin", b.Scale(gate, 3.14159265359f)) : b.Silu(gate));
        return b.Add(x, b.Linear(gated, n + "mlp.down.weight", n + "mlp.down.bias"));
    }

    // DynamicTanh (gamma · tanh(alpha · x) + beta) or an RMS norm.
    private static Value Norm(GraphBuilder b, Value x, SameDecoderParameters p, string n)
        => p.DynamicTanh
            ? b.Add(b.Mul(b.Unary("Tanh", b.Mul(x, b.Weight(n + ".alpha"))), b.Weight(n + ".weight")), b.Weight(n + ".bias"))
            : b.RmsNorm(x, b.Weight(n + ".weight"), SameDecoderParameters.NormEps);

    // Rotary positions of rows that restart every `period` rows.
    private static Value Positions(GraphBuilder b, string name, long rows, long period)
    {
        var values = new int[rows];
        for (long r = 0; r < rows; r++)
        {
            values[r] = (int)(r % period);
        }

        return b.Constant(name, DataType.I32, new Shape(rows, 1), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
    }
}
