using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// com.microsoft contrib operators beyond attention: fused linear layers, activations, normalizations, layout variants and utilities.
internal sealed partial class OnnxTranslator
{
    // Contrib operators with exactly the semantics of the default-domain operator of the same name.
    private static readonly HashSet<string> s_contribAliases = ["Trilu", "Range", "GatherND", "Pad", "QuantizeLinear", "DequantizeLinear"];

    /// <summary>A contrib node that is translated as its default-domain twin (the same node in the default domain); otherwise the node.</summary>
    private static OnnxNode Canonical(OnnxNode n)
        => n.Domain == "com.microsoft" && s_contribAliases.Contains(n.OpType)
            ? new OnnxNode { Name = n.Name, OpType = n.OpType, Domain = "", Inputs = n.Inputs, Outputs = n.Outputs, Attributes = n.Attributes }
            : n;

    /// <summary>An optional output that the graph reads.</summary>
    private static bool Wants(OnnxNode n, int i) => n.Outputs.Count > i && n.Outputs[i] != "";

    /// <summary>FusedGemm: Gemm followed by an activation (activation_alpha / _beta / _gamma as its parameters).</summary>
    private void FusedGemm(OnnxNode n, TranslatedTensor?[] ins)
    {
        var gemm = Synthetic("Gemm", [.. ins.TakeWhile(t => t is not null).Cast<TranslatedTensor>()], [.. n.Attributes.Values.Where(a => !a.Name.StartsWith("activation", StringComparison.Ordinal))]);
        float[] parameters = [n.Float("activation_alpha", n.Str("activation", "") == "LeakyRelu" ? 0.01f : 0.2f), n.Float("activation_beta", 0.5f)];
        SetDyn(n, 0, Act(n.Str("activation", ""), AsValue(gemm), parameters));
    }

    /// <summary>GemmFastGelu: X · W (+ bias) followed by the tanh approximation of Gelu.</summary>
    private void GemmFastGelu(OnnxNode n, TranslatedTensor?[] ins)
    {
        var y = AsValue(Synthetic("MatMul", [ins[0]!, ins[1]!]));
        if (Opt(ins, 2) is not null)
        {
            y = _b.Add(y, X(ins, 2));
        }

        SetDyn(n, 0, Emit("GeluTanh", [y], null, Nm(n)));
    }

    /// <summary>ScaledSiLU: silu(alpha · x), alpha from the optional scale input or the attribute.</summary>
    private void ScaledSilu(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var scaled = Opt(ins, 1) is not null ? _b.Mul(x, X(ins, 1)) : _b.Scale(x, n.Float("alpha", 1f));
        SetDyn(n, 0, Emit("Silu", [scaled], null, Nm(n)));
    }

    /// <summary>BiasSplitGelu: (x + bias) split in halves along the last axis, left · gelu(right).</summary>
    private void BiasSplitGelu(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = _b.Add(X(ins, 0), X(ins, 1));
        int last = x.Rank - 1;
        long half = x.Shape[last].Fixed / 2;
        SetDyn(n, 0, _b.Mul(_b.Slice(x, last, 0, half), Emit("Gelu", [_b.Slice(x, last, half, half)]), Nm(n)));
    }

    /// <summary>GatedRMSNorm: rms_norm(x) · scale · act(gate), act silu (swish) or sigmoid.</summary>
    private void GatedRmsNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        var y = _b.RmsNorm(X(ins, 0), X(ins, 1), n.Float("epsilon", 1e-6f));
        var gate = X(ins, 2);
        var activated = n.Str("activation", "silu") switch
        {
            "silu" or "swish" => _b.Silu(gate),
            "sigmoid" => _b.Sigmoid(gate),
            var other => throw new NotSupportedException($"GatedRMSNorm activation '{other}'"),
        };
        SetDyn(n, 0, _b.Mul(y, activated, Nm(n)));
    }

    /// <summary>
    /// FusedMatMul / TransposeMatMul: alpha · op(A) · op(B) where transBatchA/B first move the leading axis behind the batch axes
    /// ([d0, d1, ..., M, K] → [d1, ..., d0, ...]) and transA/B then swap the last two.
    /// </summary>
    private void FusedMatMul(OnnxNode n, TranslatedTensor?[] ins)
    {
        Value Operand(int i, string batchFlag)
        {
            var v = X(ins, i);
            if (n.Int(batchFlag, 0) == 0 || v.Rank < 3)
            {
                return v;
            }

            int r = v.Rank;
            return Permute(v, [.. Enumerable.Range(1, r - 2), 0, r - 1]);
        }

        var attributes = Attributes.From(("trans_a", n.Int("transA", 0) != 0), ("trans_b", n.Int("transB", 0) != 0), ("alpha", n.Float("alpha", 1f)));
        SetDyn(n, 0, Emit("BatchMatMul", [Operand(0, "transBatchA"), Operand(1, "transBatchB")], attributes, Nm(n)));
        Heavy();
    }

    /// <summary>CDist: pairwise distances between the rows of A [M, K] and B [N, K], squared Euclidean (sqeuclidean) or Euclidean.</summary>
    private void CDist(OnnxNode n, TranslatedTensor?[] ins)
    {
        var a = X(ins, 0);
        var b = X(ins, 1);
        long m = a.Shape[0].Fixed;
        long cols = b.Shape[0].Fixed;
        var all = L(-1);
        var aa = Emit("Reduce", [_b.Mul(a, a)], Attributes.From(("mode", "sum"), ("axes", all), ("keepdims", true)));
        var bb = Rs(Emit("Reduce", [_b.Mul(b, b)], Attributes.From(("mode", "sum"), ("axes", all), ("keepdims", true))), 1, cols);
        var ab = Emit("BatchMatMul", [a, b], Attributes.From(("trans_b", true), ("alpha", -2f)));
        Heavy();
        var squared = Emit("Clip", [_b.Add(_b.Add(Rs(aa, m, 1), bb), ab)], Attributes.From(("min", 0f)));
        SetDyn(n, 0, n.Str("metric", "sqeuclidean") switch
        {
            "sqeuclidean" => squared,
            "euclidean" => _b.Unary("Sqrt", squared, Nm(n)),
            var metric => throw new NotSupportedException($"CDist metric '{metric}'"),
        });
    }

    /// <summary>ComplexMul / ComplexMulConj of complex tensors stored as [..., 2] (real, imaginary), broadcasting.</summary>
    private void ComplexMul(OnnxNode n, TranslatedTensor?[] ins, bool conjugate)
    {
        var a = X(ins, 0);
        var b = X(ins, 1);
        Value Part(Value x, int i) => _b.Slice(x, x.Rank - 1, i, 1);
        var (ar, ai, br, bi) = (Part(a, 0), Part(a, 1), Part(b, 0), Part(b, 1));
        if (conjugate)
        {
            bi = _b.Scale(bi, -1f);
        }

        var real = _b.Sub(_b.Mul(ar, br), _b.Mul(ai, bi));
        var imaginary = _b.Add(_b.Mul(ar, bi), _b.Mul(ai, br));
        SetDyn(n, 0, Emit("Concat", [real, imaginary], Attributes.From(("axis", (long)Math.Max(real.Rank - 1, 0))), Nm(n)));
    }

    /// <summary>
    /// GroupNorm and SkipGroupNorm (diffusion UNet exports): group normalization of x [N, H, W, C] (channels_last) or [N, C, H, W] with
    /// gamma/beta [C] and an optional SiLU; SkipGroupNorm first adds skip ([N, 1, 1, C], [N, C] or full) and bias and also returns that sum.
    /// </summary>
    private void ContribGroupNorm(OnnxNode n, TranslatedTensor?[] ins, bool skip)
    {
        var x = X(ins, 0);
        bool channelsLast = n.Int("channels_last", 1) != 0;
        if (skip)
        {
            var s = X(ins, 3);
            if (s.Rank == 2 && x.Rank == 4)
            {
                s = channelsLast ? Rs(s, 0, 1, 1, s.Shape[1].Fixed) : Rs(s, 0, s.Shape[1].Fixed, 1, 1);
            }

            x = _b.Add(x, s);
            if (Opt(ins, 4) is not null)
            {
                x = _b.Add(x, X(ins, 4));
            }

            if (Wants(n, 1))
            {
                SetDyn(n, 1, x);
            }
        }

        int rank = x.Rank;
        var nchw = channelsLast ? Permute(x, [0, rank - 1, .. Enumerable.Range(1, rank - 2)]) : x;
        var y = Emit("GroupNorm", [nchw, X(ins, 1), X(ins, 2)], Attributes.From(("groups", n.Int("groups", 1)), ("eps", n.Float("epsilon", 1e-5f))));
        if (n.Int("activation", 0) == 1)
        {
            y = _b.Silu(y);
        }

        SetDyn(n, 0, channelsLast ? Permute(y, [0, .. Enumerable.Range(2, rank - 2), 1]) : y);
    }

    /// <summary>NhwcMaxPool: MaxPool of an [N, H, W, C] tensor.</summary>
    private void NhwcMaxPool(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = ins[0]!;
        int rank = OnnxRank(x);
        var nchw = Synthetic("Transpose", [x], IntsAttribute("perm", [0, rank - 1, .. Enumerable.Range(1, rank - 2).Select(i => (long)i)]));
        var pooled = Synthetic("MaxPool", [nchw], [.. n.Attributes.Values]);
        var y = Synthetic("Transpose", [pooled], IntsAttribute("perm", [0, .. Enumerable.Range(2, rank - 2).Select(i => (long)i), 1]));
        _env[n.Outputs[0]] = y;
    }

    /// <summary>UnfoldTensor (torch.Tensor.unfold): windows of <c>size</c> every <c>step</c> along <c>dim</c>, the window axis appended last.</summary>
    private void UnfoldTensor(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int dim = NormalizeAxis(n.Int("dim", -1), x.Rank);
        long size = n.Int("size", 1);
        long step = n.Int("step", 1);
        long length = x.Shape[dim].Fixed;
        long windows = (length - size) / step + 1;
        var index = Rs(IndexConstant(Nm(n) + ".windows", [.. Enumerable.Range(0, (int)(windows * size)).Select(i => (int)(i / size * step + i % size))]), windows, size);
        var gathered = Emit("GatherAxis", [x, index], Attributes.From(("axis", (long)dim)));
        // [..., windows, size, after...] → [..., windows, after..., size]
        int r = gathered.Rank;
        int[] perm = [.. Enumerable.Range(0, dim + 1), .. Enumerable.Range(dim + 2, r - dim - 2), dim + 1];
        SetDyn(n, 0, Permute(gathered, perm));
    }

    /// <summary>
    /// EmbedLayerNormalization (BERT exports): LayerNorm(word[ids] + position[position_ids or 0..S-1] + segment[segment_ids]); outputs the
    /// mask index (number of ones per mask row) and the embedding sum before the norm.
    /// </summary>
    private void EmbedLayerNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var ids = Idx(ins[0]!);
        long s = ids.Shape[1].Fixed;
        Value Lookup(int table, Value indices) => Emit("GatherAxis", [X(ins, table), indices], Attributes.From(("axis", 0L)));
        var sum = Lookup(2, ids);
        var positions = Opt(ins, 8) is { } positionIds ? Idx(positionIds) : IndexConstant(name + ".positions", [.. Enumerable.Range(0, (int)s)]);
        var position = Lookup(3, positions);
        sum = _b.Add(sum, position.Rank == 2 ? Rs(position, 1, s, position.Shape[1].Fixed) : position);
        if (Opt(ins, 1) is { } segments && Opt(ins, 4) is not null)
        {
            sum = _b.Add(sum, Lookup(4, Idx(segments)));
        }

        SetDyn(n, 0, Emit("LayerNorm", [sum, X(ins, 5), X(ins, 6)], Attributes.From(("eps", n.Float("epsilon", 1e-12f))), name));
        if (Wants(n, 1))
        {
            var count = Opt(ins, 7) is { } mask
                ? Emit("Reduce", [AsValue(mask)], Attributes.From(("mode", "sum"), ("axes", L(1)), ("keepdims", false)))
                : _b.Scale(Emit("Reduce", [AsValue(ins[0]!)], Attributes.From(("mode", "sum"), ("axes", L(1)), ("keepdims", false))), 0f, s);
            _env[n.Outputs[1]] = Typed(count, OnnxType.Int32);
        }

        if (Wants(n, 2))
        {
            SetDyn(n, 2, sum);
        }
    }

    /// <summary>
    /// CropAndResize (TensorFlow semantics): bilinear or nearest crops of x [N, C, H, W] at normalized boxes [R, 4] (y1, x1, y2, x2) of
    /// images batch_indices [R], resized to crop_size; samples outside the image get extrapolation_value. Implemented as GridSample with
    /// aligned corners over a grid computed from the boxes.
    /// </summary>
    private void CropAndResize(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var size = ConstIn(ins, 3, "CropAndResize crop_size").Ints();
        long ch = size[0];
        long cw = size[1];
        var images = Synthetic("Gather", [ins[0]!, ins[2]!], IntAttribute("axis", 0));
        var boxes = X(ins, 1);
        long r = boxes.Shape[0].Fixed;
        Value Coordinate(int lo, int hi, long count, bool rows)
        {
            // 2 · (lo + a_i · (hi - lo)) - 1 with a_i = i / (count - 1) (0.5 for a single sample).
            var start = _b.Slice(boxes, 1, lo, 1);
            var extent = _b.Sub(_b.Slice(boxes, 1, hi, 1), start);
            var steps = F32Const([.. Enumerable.Range(0, (int)count).Select(i => count > 1 ? (float)i / (count - 1) : 0.5f)], name + (rows ? ".ys" : ".xs"));
            var c = _b.Scale(_b.Add(start, _b.Mul(extent, Rs(steps, 1, count))), 2f, -1f);
            var zeros = _b.Constant(name + (rows ? ".yz" : ".xz"), DataType.F32, rows ? new Shape(1, 1, cw, 1) : new Shape(1, ch, 1, 1), new byte[ch * cw * 4 / (rows ? ch : cw)]);
            return _b.Add(rows ? Rs(c, r, ch, 1, 1) : Rs(c, r, 1, cw, 1), zeros);
        }

        var gy = Coordinate(0, 2, ch, true);
        var gx = Coordinate(1, 3, cw, false);
        var grid = TranslatedTensor.Dyn(Emit("Concat", [gx, gy], Attributes.From(("axis", 3L))));
        string mode = n.Str("mode", "bilinear") == "nearest" ? "nearest" : "linear";
        var sampled = AsValue(Synthetic("GridSample", [images, grid], IntAttribute("align_corners", 1), new OnnxAttribute { Name = "mode", Type = OnnxAttrType.String, S = mode }));
        Value Inside(Value g) => _b.Mul(_b.Binary("GreaterEqual", g, F32Const([-1f], name + ".lo")), _b.Binary("LessEqual", g, F32Const([1f], name + ".hi")));
        var inside = Rs(_b.Mul(Inside(gy), Inside(gx)), r, 1, ch, cw);
        SetDyn(n, 0, Emit("Where", [inside, sampled, F32Const([n.Float("extrapolation_value", 0f)], name + ".extrapolation")], null, name));
    }

    /// <summary>
    /// MoE (dense evaluation): softmax router probabilities, the top k per token (renormalized when normalize_routing_weights), and the
    /// weighted sum of every selected expert's fc2(act(fc1(x))). Expert weights are [E, out, in] (or the older [E, in, out]); activations
    /// relu, gelu, silu, identity and SwiGLU (fused interleaved or concatenated, with swiglu_limit, activation_alpha / _beta).
    /// </summary>
    private void Moe(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (n.Int("use_sparse_mixer", 0) != 0)
        {
            throw new NotSupportedException("MoE with the sparse mixer");
        }

        string name = Nm(n);
        var x = X(ins, 0);
        long hidden = x.Shape[x.Rank - 1].Fixed;
        var tokens = Rs(x, -1, hidden);
        var router = AsValue(ins[1]!);
        long experts = router.Shape[router.Rank - 1].Fixed;
        var probabilities = _b.Emit1("Softmax", [Rs(router, -1, experts)]);
        long k = n.Int("k", 1);
        var top = AsValue(Synthetic("TopK", [TranslatedTensor.Dyn(probabilities), TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [1], [k], name + ".k"))]));
        var threshold = Emit("Reduce", [top], Attributes.From(("mode", "min"), ("axes", L(-1)), ("keepdims", true)));
        var gates = _b.Mul(probabilities, _b.Binary("GreaterEqual", probabilities, threshold));
        if (n.Int("normalize_routing_weights", 0) != 0)
        {
            gates = _b.Binary("Div", gates, Emit("Reduce", [gates], Attributes.From(("mode", "sum"), ("axes", L(-1)), ("keepdims", true))));
        }

        Value Layer(int weight, int bias, Value input)
        {
            var w = ins[weight]!;
            long inputWidth = input.Shape[input.Rank - 1].Fixed;
            bool outIn = OnnxRank(w) == 3 && (w.IsConst ? w.C!.Dims[2] : w.V!.Shape[2].Fixed) == inputWidth;
            var y = Emit("BatchMatMul", [input, AsValue(w)], Attributes.From(("trans_b", outIn)));
            Heavy();
            if (Opt(ins, bias) is not null)
            {
                var b = X(ins, bias);
                y = _b.Add(y, Rs(b, experts, 1, b.Shape[1].Fixed));
            }

            return y;
        }

        var h = Layer(2, 3, Rs(tokens, 1, -1, hidden));   // [E, T, F]
        string activation = n.Str("activation_type", "relu");
        if (activation == "swiglu")
        {
            long width = h.Shape[2].Fixed / 2;
            Value gate, linear;
            if (n.Int("swiglu_fusion", 0) == 1)
            {
                var pairs = Rs(h, experts, -1, width, 2);
                gate = Rs(_b.Slice(pairs, 3, 0, 1), experts, -1, width);
                linear = Rs(_b.Slice(pairs, 3, 1, 1), experts, -1, width);
            }
            else if (n.Int("swiglu_fusion", 0) == 2)
            {
                gate = _b.Slice(h, 2, 0, width);
                linear = _b.Slice(h, 2, width, width);
            }
            else
            {
                gate = h;
                linear = Layer(6, 7, Rs(tokens, 1, -1, hidden));
            }

            if (n.Has("swiglu_limit"))
            {
                float limit = n.Float("swiglu_limit", 0f);
                gate = Emit("Clip", [gate], Attributes.From(("max", limit)));
                linear = Emit("Clip", [linear], Attributes.From(("min", -limit), ("max", limit)));
            }

            float alpha = n.Float("activation_alpha", 1.702f);
            float beta = n.Float("activation_beta", 1f);
            h = _b.Mul(_b.Mul(gate, _b.Sigmoid(_b.Scale(gate, alpha))), _b.Scale(linear, 1f, beta));
        }
        else
        {
            h = activation switch
            {
                "relu" => _b.Unary("Relu", h),
                "gelu" => _b.Unary("Gelu", h),
                "silu" => _b.Silu(h),
                "identity" => h,
                _ => throw new NotSupportedException($"MoE activation '{activation}'"),
            };
            if (Opt(ins, 6) is not null)
            {
                h = _b.Mul(h, Layer(6, 7, Rs(tokens, 1, -1, hidden)));
            }
        }

        var outputs = Layer(4, 5, h);   // [E, T, H]
        var weighted = _b.Mul(outputs, Rs(Permute(gates, [1, 0]), experts, -1, 1));
        var y = Emit("Reduce", [weighted], Attributes.From(("mode", "sum"), ("axes", L(0)), ("keepdims", false)));
        SetDyn(n, 0, Emit("Reshape", [y], Attributes.From(("shape", Fixed(x))), name));
    }

    /// <summary>
    /// GatherBlockQuantized: Gather (gather_axis) from block-quantized constant data (uint8 holding 4-bit pairs or 8-bit values) with
    /// scales per block along quantize_axis and optional zero points packed like the data (default 2^(bits-1)); dequantized at import.
    /// </summary>
    private void GatherBlockQuantized(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var data = ConstIn(ins, 0, "GatherBlockQuantized data");
        var scales = ConstIn(ins, 2, "GatherBlockQuantized scales");
        var zeroPoints = OptConst(ins, 3, "GatherBlockQuantized zero points");
        long bits = n.Int("bits", 4);
        long block = n.Int("block_size", 128);
        bool packed = bits == 4 && data.ElemType is OnnxType.UInt8 or OnnxType.Int8;
        long[] dims = [.. data.Dims];
        if (packed)
        {
            dims[^1] *= 2;
        }

        int axis = NormalizeAxis(n.Int("quantize_axis", -1), dims.Length);
        var dequantized = OnnxConst.Lazy(OnnxType.Float, dims, () =>
        {
            var raw = data.Ints();
            var scale = scales.Floats();
            var zp = zeroPoints?.Ints();
            long count = dims.Aggregate(1L, (a, b) => a * b);
            long inner = dims.Skip(axis + 1).Aggregate(1L, (a, b) => a * b);
            long extent = dims[axis];
            long blocks = (extent + block - 1) / block;
            long Unpack(long[] values, long index) => packed ? ((values[index / 2] & 0xFF) >> (int)(4 * (index & 1))) & 0xF : values[index];
            var result = new float[count];
            for (long i = 0; i < count; i++)
            {
                long outer = i / (inner * extent);
                long along = i / inner % extent;
                long blockIndex = (outer * blocks + along / block) * inner + i % inner;
                double z = zp is null ? (1L << (int)(bits - 1)) : Unpack(zp, blockIndex);
                double q = Unpack(raw, i);
                if (!packed && data.ElemType == OnnxType.Int8 && zp is null)
                {
                    z = 0;
                }

                result[i] = (float)((q - z) * scale[blockIndex]);
            }

            return ToBytes(result);
        }, name + ".dequantized");
        _env[n.Outputs[0]] = Synthetic("Gather", [TranslatedTensor.Const(dequantized), ins[1]!], IntAttribute("axis", n.Int("gather_axis", 0)));
    }

    /// <summary>
    /// NGramRepeatBlock: scores [B, V] with every token that would repeat an n-gram of input_ids [B, L] set to -inf (the token after each
    /// earlier occurrence of the last n - 1 ids).
    /// </summary>
    private void NGramRepeatBlock(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var ids = X(ins, 0);
        var scores = X(ins, 1);
        long gram = n.Int("ngram_size", 1);
        long length = ids.Shape[1].Fixed;
        long b = ids.Shape[0].Fixed;
        long vocab = scores.Shape[1].Fixed;
        long windows = length - gram + 1;
        if (windows <= 0)
        {
            SetDyn(n, 0, scores);
            return;
        }

        Value match;
        if (gram > 1)
        {
            var index = Rs(IndexConstant(name + ".windows", [.. Enumerable.Range(0, (int)(windows * (gram - 1))).Select(i => (int)(i / (gram - 1) + i % (gram - 1)))]), windows, gram - 1);
            var earlier = Emit("GatherAxis", [ids, index], Attributes.From(("axis", 1L)));                  // [B, W, n-1]
            var tail = Rs(_b.Slice(ids, 1, length - gram + 1, gram - 1), b, 1, gram - 1);                     // [B, 1, n-1]
            match = Emit("Reduce", [_b.Binary("Equal", earlier, tail)], Attributes.From(("mode", "min"), ("axes", L(2)), ("keepdims", true)));
        }
        else
        {
            match = _b.Constant(name + ".all", DataType.F32, new Shape(1, windows, 1), ToBytes([.. Enumerable.Repeat(1f, (int)windows)]));
        }

        var next = Rs(_b.Slice(ids, 1, gram - 1, windows), b, windows, 1);                                   // [B, W, 1]
        var vocabulary = Rs(F32Const([.. Enumerable.Range(0, (int)vocab).Select(v => (float)v)], name + ".vocab"), 1, 1, vocab);
        var banned = Emit("Reduce", [_b.Mul(_b.Binary("Equal", next, vocabulary), match)], Attributes.From(("mode", "max"), ("axes", L(1)), ("keepdims", false)));
        SetDyn(n, 0, Emit("Where", [banned, F32Const([float.NegativeInfinity], name + ".ninf"), scores], null, name));
    }

    /// <summary>Inverse of square matrices [..., M, M] (IR Inverse).</summary>
    private void Inverse(OnnxNode n, TranslatedTensor?[] ins) => SetDyn(n, 0, Emit("Inverse", [X(ins, 0)], null, Nm(n)));

    /// <summary>ExpandDims: Unsqueeze at the axis given by the scalar input.</summary>
    private void ExpandDims(OnnxNode n, TranslatedTensor?[] ins)
    {
        long axis = ConstIn(ins, 1, "ExpandDims axis").ScalarInt();
        _env[n.Outputs[0]] = Synthetic("Unsqueeze", [ins[0]!], IntsAttribute("axes", axis));
    }
}
