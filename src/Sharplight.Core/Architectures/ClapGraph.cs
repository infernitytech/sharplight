using System.Runtime.InteropServices;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the CLAP audio tower (the HTS-AT Swin transformer: a log-mel spectrogram stretched in time with bicubic interpolation and folded into a square image,
/// 4 × 4 patches, shifted-window attention with a learned relative position bias, patch merging, mean pooling, the two-layer projection) and the text tower (a RoBERTa
/// encoder, the tanh pooler on the first token and the projection) as graphs of generic ops. See <see cref="AudioEmbedderGraphs"/>.
/// </summary>
internal static class ClapGraph
{
    /// <summary>Builds the audio graph: <c>features</c> [frames, mel bins] to the embedding.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildAudio(ClapParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioEmbedderGraphs.AudioGraph, weights);
        var x = b.Input(AudioEmbedderGraphs.FeaturesInput, InputRole.Media, DataType.F32, new Shape(p.Frames, p.MelBins));
        long side = p.SpecSize;
        long ratio = p.FrequencyRatio;
        long width = side * ratio;
        using (b.Group("audio.input"))
        {
            // The mel batch normalization (folded into a scale and a shift per mel bin), then the time axis stretched to side · ratio steps.
            x = b.Add(b.Mul(x, b.Weight("audio.batch_norm.scale")), b.Weight("audio.batch_norm.shift"));
            if (p.Frames < width)
            {
                x = b.Emit1("BatchMatMul", [Constant(b, "audio.interpolation", BicubicMatrix(p.Frames, (int)width), (int)width, p.Frames), x]);
            }

            // [time, mel] as [ratio, side, mel] -> [ratio, mel, side] -> the image [ratio · mel, side].
            x = b.Reshape(x, ratio, side, p.MelBins);
            x = b.Permute(x, 0, 2, 1);
            x = b.Reshape(x, side, side);
            long patch = p.PatchSize;
            long grid = side / patch;
            x = b.Reshape(x, grid, patch, grid, patch);
            x = b.Permute(x, 0, 2, 1, 3);
            x = b.Reshape(x, grid * grid, patch * patch);
            x = b.Linear(x, "audio.patch_embed.weight", "audio.patch_embed.bias");
            x = TowerBlocks.LayerNorm(b, x, "audio.patch_norm", p.AudioEps);
        }

        long resolution = grid0(p);
        int channels = p.EmbedDim;
        for (int stage = 0; stage < p.Depths.Length; stage++)
        {
            long window = Math.Min(p.WindowSize, resolution);
            for (int block = 0; block < p.Depths[stage]; block++)
            {
                using var _ = b.Group($"audio.layers.{stage}.blocks.{block}");
                int shift = block % 2 == 1 && resolution > p.WindowSize ? p.WindowSize / 2 : 0;
                x = Block(b, p, weights, x, $"audio.layers.{stage}.blocks.{block}.", resolution, window, shift, channels, p.Heads[stage]);
            }

            if (stage + 1 < p.Depths.Length)
            {
                using var _ = b.Group($"audio.layers.{stage}.merge");
                x = Merge(b, p, x, $"audio.layers.{stage}.", resolution, channels);
                resolution /= 2;
                channels *= 2;
            }
        }

        long tokens = resolution * resolution;
        using (b.Group("audio.output"))
        {
            x = TowerBlocks.LayerNorm(b, x, "audio.norm", p.AudioEps);
            var ones = new float[tokens];
            Array.Fill(ones, 1f);
            var mean = b.Emit1("BatchMatMul", [Constant(b, "audio.pool", ones, 1, (int)tokens), x], Attributes.From(("alpha", 1f / tokens)));
            var hidden = b.Linear(mean, "audio.projection.1.weight", "audio.projection.1.bias");
            hidden = b.Unary("Relu", hidden);
            var embedding = b.Linear(hidden, "audio.projection.2.weight", "audio.projection.2.bias");
            b.Output(AudioEmbedderGraphs.EmbeddingOutput, embedding);
            b.Output(AudioEmbedderGraphs.LogitScaleOutput, b.Unary("Exp", b.Reshape(b.Weight("logit_scale_audio"), 1)));
        }

        return b.Build();

        static long grid0(ClapParameters q) => q.SpecSize / q.PatchSize;
    }

    /// <summary>Builds the text graph: <c>tokens</c> and <c>positions</c> [T] to the embedding.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildText(ClapParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioEmbedderGraphs.TextGraph, weights);
        var rows = Dim.Sym(WellKnown.TokenSymbol);
        var tokens = b.Input(AudioEmbedderGraphs.TokensInput, InputRole.TokenIds, DataType.I32, new Shape(rows));
        var positions = b.Input(AudioEmbedderGraphs.PositionsInput, InputRole.Other, DataType.I32, new Shape(rows));
        int hidden = p.TextHidden;
        int headSize = hidden / p.TextHeads;
        Value x;
        using (b.Group("text.embedding"))
        {
            x = b.Add(b.Gather(b.Weight("text.token_embedding.weight"), tokens), b.Gather(b.Weight("text.position_embedding.weight"), positions));
            x = b.Add(x, b.Slice(b.Weight("text.token_type_embedding.weight"), 0, 0, 1));
            x = TowerBlocks.LayerNorm(b, x, "text.embedding_norm", p.TextEps);
        }

        for (int il = 0; il < p.TextLayers; il++)
        {
            using var _ = b.Group($"text.layers.{il}");
            string n = $"text.layers.{il}.";
            var q = b.Reshape(b.Linear(x, n + "attention.q.weight", n + "attention.q.bias"), 0, p.TextHeads, headSize);
            var k = b.Reshape(b.Linear(x, n + "attention.k.weight", n + "attention.k.bias"), 0, p.TextHeads, headSize);
            var v = b.Reshape(b.Linear(x, n + "attention.v.weight", n + "attention.v.bias"), 0, p.TextHeads, headSize);
            var attention = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(headSize)), ("causal", false)));
            var o = b.Linear(b.Reshape(attention, 0, hidden), n + "attention.o.weight", n + "attention.o.bias");
            x = TowerBlocks.LayerNorm(b, b.Add(x, o), n + "attention_norm", p.TextEps);
            var mlp = b.Linear(b.Unary("Gelu", b.Linear(x, n + "mlp.up.weight", n + "mlp.up.bias")), n + "mlp.down.weight", n + "mlp.down.bias");
            x = TowerBlocks.LayerNorm(b, b.Add(x, mlp), n + "mlp_norm", p.TextEps);
        }

        using (b.Group("text.output"))
        {
            var first = b.Slice(x, 0, 0, 1);
            var pooled = b.Unary("Tanh", b.Linear(first, "text.pooler.weight", "text.pooler.bias"));
            var h = b.Unary("Relu", b.Linear(pooled, "text.projection.1.weight", "text.projection.1.bias"));
            b.Output(AudioEmbedderGraphs.EmbeddingOutput, b.Linear(h, "text.projection.2.weight", "text.projection.2.bias"));
            b.Output(AudioEmbedderGraphs.LogitScaleOutput, b.Unary("Exp", b.Reshape(b.Weight("logit_scale_text"), 1)));
        }

        return b.Build();
    }

    // One Swin block: pre-norm window attention (cyclically shifted for odd blocks, masked so that wrapped-around tokens do not attend to each other) and the MLP.
    private static Value Block(GraphBuilder b, ClapParameters p, Func<string, WeightInfo?> weights, Value x, string n, long resolution, long window, int shift, int channels, int heads)
    {
        long windows = resolution / window;
        long count = windows * windows;
        long cells = window * window;
        int headSize = channels / heads;
        var shortcut = x;
        var h = TowerBlocks.LayerNorm(b, x, n + "norm1", p.AudioEps);
        h = b.Reshape(h, resolution, resolution, channels);
        if (shift > 0)
        {
            h = Roll(b, h, resolution, shift);
        }

        h = b.Reshape(h, windows, window, windows, window, channels);
        h = b.Permute(h, 0, 2, 1, 3, 4);
        h = b.Reshape(h, count, cells, channels);
        Value Heads(string name)
        {
            var projected = b.Linear(h, n + name + ".weight", n + name + ".bias");
            return b.Permute(b.Reshape(projected, count, cells, heads, headSize), 0, 2, 1, 3);
        }

        var q = Heads("q");
        var k = Heads("k");
        var v = Heads("v");
        var scores = b.Emit1("BatchMatMul", [q, k], Attributes.From(("trans_b", true), ("alpha", 1f / MathF.Sqrt(headSize))));
        // The bias of every query / key cell pair of a window and head, expanded from the learned table at build time (rows of the table are too short for the GPU gather).
        var bias = Constant(b, n + "relative_bias_expanded", ExpandBias(WeightConversions.ReadFloats(weights(n + "relative_bias")!), (int)window, heads), heads, cells, cells);
        scores = b.Add(scores, bias);
        if (shift > 0)
        {
            scores = b.Add(scores, Constant(b, n + "shift_mask", ShiftMask((int)resolution, (int)window, shift), count, 1, cells, cells));
        }

        var probabilities = b.Emit1("Softmax", [scores]);
        var context = b.Emit1("BatchMatMul", [probabilities, v]);
        context = b.Reshape(b.Permute(context, 0, 2, 1, 3), count, cells, channels);
        var o = b.Linear(context, n + "o.weight", n + "o.bias");
        o = b.Reshape(o, windows, windows, window, window, channels);
        o = b.Permute(o, 0, 2, 1, 3, 4);
        o = b.Reshape(o, resolution, resolution, channels);
        if (shift > 0)
        {
            o = Roll(b, o, resolution, resolution - shift);
        }

        x = b.Add(shortcut, b.Reshape(o, resolution * resolution, channels));
        var m = TowerBlocks.LayerNorm(b, x, n + "norm2", p.AudioEps);
        m = b.Linear(b.Unary("Gelu", b.Linear(m, n + "mlp.up.weight", n + "mlp.up.bias")), n + "mlp.down.weight", n + "mlp.down.bias");
        return b.Add(x, m);
    }

    // Patch merging: the four cells of every 2 × 2 neighbourhood side by side (the column-major order of the reference), normalized and reduced to twice the channels.
    private static Value Merge(GraphBuilder b, ClapParameters p, Value x, string n, long resolution, int channels)
    {
        long half = resolution / 2;
        var grid = b.Reshape(x, half, 2, half, 2, channels);
        var parts = new Value[4];
        for (int col = 0; col < 2; col++)
        {
            for (int row = 0; row < 2; row++)
            {
                var cell = b.Slice(b.Slice(grid, 1, row, 1), 3, col, 1);
                parts[(col * 2) + row] = b.Reshape(cell, half * half, channels);
            }
        }

        var merged = b.Concat(1, parts);
        merged = TowerBlocks.LayerNorm(b, merged, n + "merge_norm", p.AudioEps);
        return b.Linear(merged, n + "merge.weight");
    }

    // torch.roll(x, -offset) over both spatial axes of [R, R, C]: the first offset rows (columns) move to the end.
    private static Value Roll(GraphBuilder b, Value x, long resolution, long offset)
    {
        for (int axis = 0; axis < 2; axis++)
        {
            x = b.Concat(axis, b.Slice(x, axis, offset, resolution - offset), b.Slice(x, axis, 0, offset));
        }

        return x;
    }

    private static Value Constant(GraphBuilder b, string name, float[] values, params long[] shape)
        => b.Constant(name, DataType.F32, new Shape([.. shape.Select(Dim.Of)]), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    private static Value Constant(GraphBuilder b, string name, float[] values, int rows, int columns) => Constant(b, name, values, (long)rows, columns);

    // [heads, w², w²]: table[index[i, j], head] for every query cell i and key cell j.
    internal static float[] ExpandBias(float[] table, int window, int heads)
    {
        int cells = window * window;
        var index = RelativeIndex(window);
        var bias = new float[heads * cells * cells];
        for (int h = 0; h < heads; h++)
        {
            for (int ij = 0; ij < cells * cells; ij++)
            {
                bias[(h * cells * cells) + ij] = table[(index[ij] * heads) + h];
            }
        }

        return bias;
    }

    // Row i · w² + j: the row of the bias table for the query cell i and the key cell j of a window (Swin's relative_position_index).
    internal static int[] RelativeIndex(int window)
    {
        int cells = window * window;
        var index = new int[cells * cells];
        for (int i = 0; i < cells; i++)
        {
            for (int j = 0; j < cells; j++)
            {
                int dh = (i / window) - (j / window) + window - 1;
                int dw = (i % window) - (j % window) + window - 1;
                index[(i * cells) + j] = (dh * ((2 * window) - 1)) + dw;
            }
        }

        return index;
    }

    // The additive mask [windows, 1, w², w²] of a shifted layout: cells that came from different regions of the unshifted image get -100.
    internal static float[] ShiftMask(int resolution, int window, int shift)
    {
        int windows = resolution / window;
        int cells = window * window;
        var region = new int[resolution];
        for (int i = 0; i < resolution; i++)
        {
            region[i] = (i >= resolution - window ? 1 : 0) + (i >= resolution - shift ? 1 : 0);
        }

        var mask = new float[windows * windows * cells * cells];
        for (int wr = 0; wr < windows; wr++)
        {
            for (int wc = 0; wc < windows; wc++)
            {
                int offset = ((wr * windows) + wc) * cells * cells;
                for (int i = 0; i < cells; i++)
                {
                    int ri = (region[(wr * window) + (i / window)] * 3) + region[(wc * window) + (i % window)];
                    for (int j = 0; j < cells; j++)
                    {
                        int rj = (region[(wr * window) + (j / window)] * 3) + region[(wc * window) + (j % window)];
                        mask[offset + (i * cells) + j] = ri == rj ? 0f : -100f;
                    }
                }
            }
        }

        return mask;
    }

    // The matrix [outputs, inputs] of torch's bicubic interpolation (align_corners, A = -0.75) along one axis.
    internal static float[] BicubicMatrix(int inputs, int outputs)
    {
        const double A = -0.75;
        var matrix = new float[outputs * inputs];
        double scale = outputs > 1 ? (inputs - 1.0) / (outputs - 1.0) : 0.0;
        for (int o = 0; o < outputs; o++)
        {
            double source = o * scale;
            int floor = (int)Math.Floor(source);
            double t = source - floor;
            double[] coefficients =
            [
                (((A * (t + 1)) - (5 * A)) * (t + 1) + (8 * A)) * (t + 1) - (4 * A),
                (((A + 2) * t) - (A + 3)) * t * t + 1,
                (((A + 2) * (1 - t)) - (A + 3)) * (1 - t) * (1 - t) + 1,
                (((A * (2 - t)) - (5 * A)) * (2 - t) + (8 * A)) * (2 - t) - (4 * A),
            ];
            for (int tap = 0; tap < 4; tap++)
            {
                int index = Math.Clamp(floor - 1 + tap, 0, inputs - 1);
                matrix[(o * inputs) + index] += (float)coefficients[tap];
            }
        }

        return matrix;
    }
}
