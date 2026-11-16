using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The prompt stage of <see cref="Sam2Adapter"/> (<see cref="WellKnown.SegmentDecodeGraph"/>): the prompt encoder (Fourier embeddings of the points and box
/// corners, learned kind embeddings, the no-mask dense embedding), the two-way transformer over [object, IoU and mask tokens | prompt tokens] and the image
/// embedding, the upscaling with the two skip maps, the hypernetwork mask heads, the IoU head and the object presence head.
/// </summary>
internal static class Sam2DecoderGraph
{
    private const float DecoderEps = 1e-5f;

    /// <summary>Builds the graph.</summary>
    /// <param name="src">The model source (weights and hyper-parameters).</param>
    /// <param name="p">The hyper-parameters.</param>
    public static ComputeGraph Build(ModelSource src, Sam2Parameters p)
    {
        var b = new GraphBuilder(WellKnown.SegmentDecodeGraph, src.Weight);
        var k = new ConstantCache(b);
        int grid = p.ImageSize / p.PromptPatch;
        int c = p.NeckSize;
        int cells = grid * grid;
        var points = Dim.Sym(WellKnown.SegmentPointSymbol);
        var embedding = b.Input(WellKnown.SegmentEmbeddingOutput, InputRole.Media, DataType.F32, new Shape(cells, c));
        var features0 = b.Input(WellKnown.SegmentFeatures0Output, InputRole.Media, DataType.F32, new Shape(1, c / 8, grid * 4, grid * 4));
        var features1 = b.Input(WellKnown.SegmentFeatures1Output, InputRole.Media, DataType.F32, new Shape(1, c / 4, grid * 2, grid * 2));
        var coords = b.Input("point_coords", InputRole.Other, DataType.F32, new Shape(points, 2));
        var kinds = b.Input("point_kinds", InputRole.Other, DataType.I32, new Shape(points));

        Value tokens;
        Value keys;
        Value keyPositions;
        using (b.Group("prompt"))
        {
            var sparse = SparsePrompt(b, k, p, coords, kinds);
            var outputTokens = b.Concat(0, b.Weight("mask_decoder.obj_score_token.weight"), b.Weight("mask_decoder.iou_token.weight"), b.Weight("mask_decoder.mask_tokens.weight"));
            tokens = b.Concat(0, outputTokens, sparse);
            keys = b.Add(embedding, b.Reshape(b.Weight("prompt_encoder.no_mask_embed.weight"), c));
            keyPositions = GridPositions(b, k, grid);
        }

        var queries = tokens;
        for (int il = 0; il < p.DecoderLayers; il++)
        {
            using var _ = b.Group($"decoder.layers.{il}");
            string w = $"mask_decoder.transformer.layers.{il}.";
            if (il == 0)
            {
                queries = Attention(b, w + "self_attn", queries, queries, queries, p.DecoderHeads, c);
            }
            else
            {
                var withPosition = b.Add(queries, tokens);
                queries = b.Add(queries, Attention(b, w + "self_attn", withPosition, withPosition, queries, p.DecoderHeads, c));
            }

            queries = LayerNorm(b, queries, w + "layer_norm1");
            var queryPosition = b.Add(queries, tokens);
            var keyPosition = b.Add(keys, keyPositions);
            queries = LayerNorm(b, b.Add(queries, Attention(b, w + "cross_attn_token_to_image", queryPosition, keyPosition, keys, p.DecoderHeads, c / p.DecoderDownsample)), w + "layer_norm2");
            queries = LayerNorm(b, b.Add(queries, Mlp(b, w + "mlp", queries, 2)), w + "layer_norm3");
            queryPosition = b.Add(queries, tokens);
            keys = LayerNorm(b, b.Add(keys, Attention(b, w + "cross_attn_image_to_token", keyPosition, queryPosition, queries, p.DecoderHeads, c / p.DecoderDownsample)), w + "layer_norm4");
        }

        using (b.Group("decoder.final"))
        {
            var final = Attention(b, "mask_decoder.transformer.final_attn_token_to_image", b.Add(queries, tokens), b.Add(keys, keyPositions), keys, p.DecoderHeads, c / p.DecoderDownsample);
            queries = LayerNorm(b, b.Add(queries, final), "mask_decoder.transformer.layer_norm_final_attn");
        }

        using (b.Group("masks"))
        {
            int maskTokens = p.MaskTokens;
            var map = b.Reshape(b.Contiguous(b.Permute(keys, 1, 0)), 1, c, grid, grid);
            var up1 = b.Add(Transposed(b, map, "mask_decoder.upscale_conv1"), features1);
            var norm = b.Permute(b.Emit1("LayerNorm", [b.Contiguous(b.Permute(up1, 0, 2, 3, 1)), b.Weight("mask_decoder.upscale_layer_norm.weight"), b.Weight("mask_decoder.upscale_layer_norm.bias")],
                Attributes.From(("eps", 1e-6f))), 0, 3, 1, 2);
            var up2 = b.Unary("Gelu", b.Add(Transposed(b, b.Unary("Gelu", b.Contiguous(norm)), "mask_decoder.upscale_conv2"), features0));
            int side = grid * 4;
            var upscaled = b.Reshape(up2, c / 8, (long)side * side);

            var hyper = new Value[maskTokens];
            for (int i = 0; i < maskTokens; i++)
            {
                hyper[i] = Mlp(b, $"mask_decoder.output_hypernetworks_mlps.{i}", Pick(b, k, queries, 2 + i), 3);
            }

            var masks = b.Emit1("BatchMatMul", [b.Concat(0, hyper), upscaled]);
            b.Output(WellKnown.SegmentMasksOutput, b.Reshape(masks, maskTokens, side, side));
            var iou = b.Sigmoid(Mlp(b, "mask_decoder.iou_prediction_head", Pick(b, k, queries, 1), 3));
            b.Output(WellKnown.SegmentScoresOutput, b.Reshape(iou, maskTokens));
            b.Output(WellKnown.SegmentObjectOutput, b.Reshape(Mlp(b, "mask_decoder.pred_obj_score_head", Pick(b, k, queries, 0), 3), 1));
        }

        return b.Build();
    }

    /// <summary>Fourier embedding of the prompt coordinates (pixels of the model input, shifted to the pixel centre) plus the learned embedding of each point's kind.</summary>
    private static Value SparsePrompt(GraphBuilder b, ConstantCache k, Sam2Parameters p, Value coords, Value kinds)
    {
        int c = p.NeckSize;
        float size = p.ImageSize;
        var normalized = b.Scale(coords, 2f / size, 1f / size - 1f);
        var fourier = Fourier(b, normalized);
        var table = b.Concat(0, b.Weight("prompt_encoder.point_embed.weight"), b.Weight("prompt_encoder.not_a_point_embed.weight"));      // rows: negative, positive, box start, box end, none
        var gates = k.F32("prompt_gates", new Shape(5, c), () =>
        {
            var g = new float[5 * c];
            Array.Fill(g, 1f, 0, 4 * c);
            return g;
        });
        return b.Add(b.Mul(fourier, b.Gather(gates, kinds)), b.Gather(table, kinds));
    }

    /// <summary>The Fourier features [N, C] of coordinates [N, 2] in [−1, 1]: sin and cos of 2π · coords · the shared random matrix.</summary>
    private static Value Fourier(GraphBuilder b, Value normalized)
    {
        var angles = b.Scale(b.Emit1("BatchMatMul", [normalized, b.Weight("shared_image_embedding.positional_embedding")]), 2f * MathF.PI);
        return b.Concat(1, b.Unary("Sin", angles), b.Unary("Cos", angles));
    }

    /// <summary>The positional embedding of the cells of the image embedding grid, rows [grid², C] in raster order.</summary>
    private static Value GridPositions(GraphBuilder b, ConstantCache k, int grid)
    {
        var coords = k.F32($"grid_coords_{grid}", new Shape((long)grid * grid, 2), () =>
        {
            var v = new float[grid * grid * 2];
            for (int y = 0; y < grid; y++)
            {
                for (int x = 0; x < grid; x++)
                {
                    v[(y * grid + x) * 2] = 2f * ((x + 0.5f) / grid) - 1f;
                    v[(y * grid + x) * 2 + 1] = 2f * ((y + 0.5f) / grid) - 1f;
                }
            }

            return v;
        });
        return Fourier(b, coords);
    }

    /// <summary>An attention layer with its own projections (<c>q_proj</c>, <c>k_proj</c>, <c>v_proj</c>, <c>o_proj</c>) over an inner width.</summary>
    private static Value Attention(GraphBuilder b, string name, Value q, Value key, Value value, int heads, int inner)
    {
        int d = inner / heads;
        var qh = b.Reshape(b.Linear(q, name + ".q_proj.weight", name + ".q_proj.bias"), Rows(q), heads, d);
        var kh = b.Reshape(b.Linear(key, name + ".k_proj.weight", name + ".k_proj.bias"), Rows(key), heads, d);
        var vh = b.Reshape(b.Linear(value, name + ".v_proj.weight", name + ".v_proj.bias"), Rows(value), heads, d);
        var attn = b.Emit1("Attention", [qh, kh, vh], Attributes.From(("scale", 1f / MathF.Sqrt(d)), ("causal", false)));
        return b.Linear(b.Reshape(attn, Rows(q), inner), name + ".o_proj.weight", name + ".o_proj.bias");
    }

    // One row of a value whose row count is symbolic (a slice of the symbolic axis at an offset is not supported).
    private static Value Pick(GraphBuilder b, ConstantCache k, Value x, int row) => b.Gather(x, k.I32($"pick_{row}", () => [row]));

    // 0 keeps a symbolic row count; fixed counts are spelled out.
    private static long Rows(Value v) => v.Shape[0].IsSymbolic ? 0 : v.Shape[0].Fixed;

    /// <summary>A ReLU MLP of <paramref name="layers"/> linear layers (<c>proj_in</c>, <c>layers.i</c>, <c>proj_out</c>).</summary>
    private static Value Mlp(GraphBuilder b, string name, Value x, int layers)
    {
        var h = b.Unary("Relu", b.Linear(x, name + ".proj_in.weight", name + ".proj_in.bias"));
        for (int i = 0; i < layers - 2; i++)
        {
            h = b.Unary("Relu", b.Linear(h, $"{name}.layers.{i}.weight", $"{name}.layers.{i}.bias"));
        }

        return b.Linear(h, name + ".proj_out.weight", name + ".proj_out.bias");
    }

    private static Value Transposed(GraphBuilder b, Value x, string name)
        => b.Emit1("ConvTranspose", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("strides", new long[] { 2, 2 }), ("pads", new long[] { 0, 0, 0, 0 })));

    private static Value LayerNorm(GraphBuilder b, Value x, string name)
        => b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", DecoderEps)));
}
