namespace Sharplight.Import.Onnx;

// com.microsoft QMoE with integer weights: dequantized at import, then translated as MoE.
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// QMoE (quant_type "int"): expert weights [E, N, K·bits/8] packed low bits first (2, 4 or 8 bits, per projection overridable),
    /// scales per row [E, N] or per block [E, N, K / block_size], zero points packed like the weights (default 2^(bits-1)). The weights
    /// are dequantized to f32 [E, N, K] and the node runs as MoE with normalized routing weights (as onnxruntime computes QMoE); router_weights, the FP4 / FP8 types and prepacked layouts are not supported.
    /// </summary>
    private void QuantizedMoe(OnnxNode n, TranslatedTensor?[] ins)
    {
        string type = n.Str("quant_type", "int");
        if (type != "int" || Opt(ins, 14) is not null || n.Int("weights_prepacked", 0) > 0)
        {
            throw new NotSupportedException($"QMoE with quant_type '{type}', router_weights or prepacked weights");
        }

        string name = Nm(n);
        long bits = n.Int("expert_weight_bits", 4);
        long block = n.Int("block_size", 0);
        TranslatedTensor? Weights(int weight, int scale, int zeroPoint, string projection)
        {
            if (Opt(ins, weight) is null)
            {
                return null;
            }

            long width = n.Int(projection + "_expert_weight_bits", bits);
            var w = DequantizeExperts(ConstIn(ins, weight, $"QMoE {projection} weights"), ConstIn(ins, scale, $"QMoE {projection} scales"),
                OptConst(ins, zeroPoint, $"QMoE {projection} zero points"), width, block, name + "." + projection);
            return TranslatedTensor.Const(w);
        }

        TranslatedTensor?[] moe =
        [
            ins[0], ins[1], Weights(2, 3, 11, "fc1"), Opt(ins, 4), Weights(5, 6, 12, "fc2"), Opt(ins, 7), Weights(8, 9, 13, "fc3"), Opt(ins, 10),
        ];
        // onnxruntime's QMoE renormalizes the k selected routing weights whatever normalize_routing_weights says.
        var attributes = new Dictionary<string, OnnxAttribute>(n.Attributes) { ["normalize_routing_weights"] = IntAttribute("normalize_routing_weights", 1) };
        Moe(new OnnxNode { Name = n.Name, OpType = "MoE", Domain = n.Domain, Inputs = n.Inputs, Outputs = n.Outputs, Attributes = attributes }, moe);
    }

    /// <summary>(q - zp) · scale of packed expert weights [E, N, K·bits/8] → f32 [E, N, K].</summary>
    private static OnnxConst DequantizeExperts(OnnxConst packed, OnnxConst scales, OnnxConst? zeroPoints, long bits, long block, string name)
    {
        if (bits is not (2 or 4 or 8) || packed.Rank != 3)
        {
            throw new NotSupportedException($"QMoE with {bits}-bit weights of rank {packed.Rank}");
        }

        long experts = packed.Dims[0];
        long rows = packed.Dims[1];
        long perByte = 8 / bits;
        long k = packed.Dims[2] * perByte;
        long blocks = block > 0 ? (k + block - 1) / block : 1;
        long mask = (1L << (int)bits) - 1;
        var q = packed.Floats();
        var s = scales.Floats();
        var z = zeroPoints?.Floats();
        // Zero points are packed per row like the weights: [E, N·bits/8] per row, or [E, N, ceil(blocks·bits/8)] per block.
        long zRow = block > 0 ? (blocks * bits + 7) / 8 : 0;
        double ZeroPoint(long e, long r, long b)
        {
            if (z is null)
            {
                return 1L << (int)(bits - 1);
            }

            long index = block > 0 ? b : r;
            long offset = block > 0 ? (e * rows + r) * zRow : e * ((rows * bits + 7) / 8);
            long word = (long)z[offset + index / perByte];
            return (word >> (int)(index % perByte * bits)) & mask;
        }

        var values = new double[experts * rows * k];
        for (long e = 0; e < experts; e++)
        {
            for (long r = 0; r < rows; r++)
            {
                long row = e * rows + r;
                for (long c = 0; c < k; c++)
                {
                    long word = (long)q[row * packed.Dims[2] + c / perByte];
                    long value = (word >> (int)(c % perByte * bits)) & mask;
                    long b = block > 0 ? c / block : 0;
                    double scale = s[row * blocks + b];
                    values[row * k + c] = (float)((value - ZeroPoint(e, r, b)) * scale);
                }
            }
        }

        return OnnxConst.FromFloats(OnnxType.Float, [experts, rows, k], values, name);
    }
}
