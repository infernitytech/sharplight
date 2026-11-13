using System.Runtime.InteropServices;
using System.Text;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// ScatterND with compile-time indices (e.g. exported fold / patch-merge loops: chains of dozens of scatters into one
/// tensor) is lowered as one gather kernel per chain: a constant map gives, for every output element, the update
/// element that writes it last (or "keep the data element"). Runtime indices use a parallel scatter kernel.
/// </summary>
public sealed partial class GraphLowerer
{
    /// <summary>Update tensors (distinct buffers) one gather kernel can read; longer chains are split into in-place passes.</summary>
    private const int ScatterGatherMaxSources = 8;

    /// <summary>Largest output handled by the gather form (the map costs 4 bytes per element; one dispatch dimension).</summary>
    private const long ScatterGatherMaxElements = 65535L * 256;

    /// <summary>Map entry meaning "keep the data element"; others are (source slot &lt;&lt; 28) | element offset.</summary>
    private const uint ScatterKeep = 0xFFFFFFFFu;
    private const int ScatterSlotShift = 28;
    private const long ScatterMaxOffset = (1L << ScatterSlotShift) - 1;

    /// <summary>Last ScatterND of a folded chain → every scatter of the chain in execution order.</summary>
    private readonly Dictionary<Node, List<Node>> _scatterChains = [];

    /// <summary>
    /// Folds chains of ScatterND with constant indices (each intermediate result only feeds the next scatter) into the
    /// unit of the last scatter, and drops dead views of the absorbed intermediates (exporters leave such slices behind).
    /// </summary>
    private void FoldScatterChains(List<LoweringUnit> units, Dictionary<Value, List<Node>> consumers)
    {
        var dead = DeadViews(consumers);
        var chainUnit = new Dictionary<Node, LoweringUnit>();
        var hidden = new HashSet<Value>();
        var removed = new HashSet<LoweringUnit>();
        foreach (var unit in units)
        {
            if (unit.Node is not { } node)
            {
                continue;
            }

            if (unit.IsView && dead.Contains(node) && hidden.Contains(node.Inputs[0]))
            {
                hidden.UnionWith(node.Outputs);
                removed.Add(unit);
                continue;
            }

            if (!IsGatherableScatter(node))
            {
                continue;
            }

            var data = node.Inputs[0];
            var chain = new List<Node>();
            if (data.Producer is { } previous && chainUnit.TryGetValue(previous, out var previousUnit)
                && !_segmentOutputs.Contains(data) && consumers[data].All(c => c == node || dead.Contains(c)))
            {
                chain.AddRange(_scatterChains[previous]);
                _scatterChains.Remove(previous);
                removed.Add(previousUnit);
                hidden.Add(data);
            }

            chain.Add(node);
            _scatterChains[node] = chain;
            chainUnit[node] = unit;
            unit.Absorbed = chain.Count > 1 ? chain[..^1] : null;
        }

        units.RemoveAll(removed.Contains);
    }

    /// <summary>View nodes whose results are never used (not segment outputs, every reader is itself a dead view).</summary>
    private HashSet<Node> DeadViews(Dictionary<Value, List<Node>> consumers)
    {
        var dead = new HashSet<Node>();
        for (int i = _seg.Nodes.Count - 1; i >= 0; i--)
        {
            var n = _seg.Nodes[i];
            if (n.Op.Category != OpCategory.View)
            {
                continue;
            }

            bool unused = n.Outputs.All(o => !_segmentOutputs.Contains(o)
                && (!consumers.TryGetValue(o, out var readers) || readers.All(dead.Contains)));
            if (unused)
            {
                dead.Add(n);
            }
        }

        return dead;
    }

    private static bool IsGatherableScatter(Node n)
    {
        if (n.Op.Name != "ScatterND" || n.Attributes.Get("reduction", "none") != "none" || n.Attributes.Get("batch_dims", 0L) != 0)
        {
            return false;
        }

        var ix = n.Inputs[1];
        bool staticShapes = n.Inputs.All(v => v.Shape.Symbol is null) && n.Output.Shape.Symbol is null;
        return ix.Kind == ValueKind.Constant && ix.ConstantData is not null && staticShapes
            && n.Output.Shape.Elements() <= ScatterGatherMaxElements;
    }

    /// <summary>
    /// Gather lowering of a folded chain, else the parallel kernel for scatters without reduction; false leaves the
    /// node to the sequential kernel (reductions).
    /// </summary>
    private bool TryLowerScatterNDFast(Node n)
    {
        if (_scatterChains.TryGetValue(n, out var chain))
        {
            if (TryLowerScatterGather(chain))
            {
                return true;
            }

            // The map does not fit: the absorbed scatters run one by one again.
            _scatterChains.Remove(n);
            foreach (var earlier in chain[..^1])
            {
                LowerScatterND(earlier);
            }
        }

        if (n.Attributes.Get("reduction", "none") != "none")
        {
            return false;
        }

        LowerScatterNDParallel(n);
        return true;
    }

    /// <summary>Lowers a folded chain (or a single constant-index scatter) as gather kernels; false when the map does not fit.</summary>
    private bool TryLowerScatterGather(List<Node> chain)
    {
        var data = Ref(chain[0].Inputs[0]);
        var outputShape = BoundShape(chain[^1].Output.Shape);
        var outputStrides = TensorRef.DenseStrides(outputShape);
        long total = outputShape.Aggregate(1L, (a, b) => a * b);
        int elementSize = data.Type.ElementSize;

        // Plan the passes: consecutive scatters while their update buffers fit the source slots.
        var passes = new List<ScatterGatherPass>();
        var current = new ScatterGatherPass();
        foreach (var node in chain)
        {
            var update = Ref(node.Inputs[2]);
            if (update.ByteOffset % elementSize != 0)
            {
                return false;
            }

            if (!current.Buffers.Contains(update.Buffer) && current.Buffers.Count == ScatterGatherMaxSources)
            {
                passes.Add(current);
                current = new ScatterGatherPass();
            }

            current.Add(node, update);
        }
        passes.Add(current);

        var maps = new List<uint[]>();
        foreach (var pass in passes)
        {
            var map = new uint[total];
            Array.Fill(map, ScatterKeep);
            for (int i = 0; i < pass.Nodes.Count; i++)
            {
                if (!FillScatterMap(map, pass.Nodes[i], pass.Updates[i], pass, outputShape, outputStrides))
                {
                    return false;
                }
            }

            maps.Add(map);
        }

        var y = Produce(chain[^1].Output);
        for (int p = 0; p < passes.Count; p++)
        {
            EmitScatterGather(chain, passes[p], maps[p], p == 0 ? data : null, y, total);
        }

        return true;
    }

    /// <summary>Writes the map entries of one scatter (later scatters overwrite earlier ones, as in sequential execution).</summary>
    private static bool FillScatterMap(uint[] map, Node node, TensorRef update, ScatterGatherPass pass, long[] outputShape, long[] outputStrides)
    {
        var ix = node.Inputs[1];
        long[] indices = ConstantIndices(ix);
        int q = (int)ix.Shape[-1].Fixed;
        long rows = indices.Length / q;
        int tailRank = outputShape.Length - q;
        long tail = 1;
        for (int k = q; k < outputShape.Length; k++)
        {
            tail *= outputShape[k];
        }

        int slot = pass.Buffers.IndexOf(update.Buffer);
        long baseElement = pass.BaseOffsets[slot] / update.Type.ElementSize;
        long updateStart = update.ByteOffset / update.Type.ElementSize - baseElement;
        var updateCoords = new long[update.Rank];
        for (long m = 0; m < rows; m++)
        {
            long destination = 0;
            bool inRange = true;
            for (int j = 0; j < q; j++)
            {
                long index = indices[m * q + j];
                index = index < 0 ? index + outputShape[j] : index;
                if (index < 0 || index >= outputShape[j])
                {
                    inRange = false;
                    break;
                }

                destination += index * outputStrides[j];
            }

            if (!inRange)
            {
                continue;
            }

            for (long t = 0; t < tail; t++)
            {
                // Update coordinates: the index-row coordinates of m, then the tail coordinates of t.
                long rest = m;
                for (int k = update.Rank - tailRank - 1; k >= 0; k--)
                {
                    updateCoords[k] = rest % update.Shape[k];
                    rest /= update.Shape[k];
                }

                rest = t;
                for (int k = update.Rank - 1; k >= update.Rank - tailRank; k--)
                {
                    updateCoords[k] = rest % update.Shape[k];
                    rest /= update.Shape[k];
                }

                long source = updateStart;
                for (int k = 0; k < update.Rank; k++)
                {
                    source += updateCoords[k] * update.Strides[k];
                }

                if (source < 0 || source > ScatterMaxOffset)
                {
                    return false;
                }

                map[destination + t] = ((uint)slot << ScatterSlotShift) | (uint)source;
            }
        }

        return true;
    }

    private static long[] ConstantIndices(Value ix)
    {
        var bytes = ix.ConstantData!;
        return ix.Type == DataType.I64
            ? MemoryMarshal.Cast<byte, long>(bytes).ToArray()
            : [.. MemoryMarshal.Cast<byte, int>(bytes).ToArray().Select(i => (long)i)];
    }

    /// <summary>
    /// One gather pass: y[e] = source[map[e]] for mapped elements, else data[e] (first pass) or unchanged (later passes,
    /// which update y in place).
    /// </summary>
    private void EmitScatterGather(List<Node> chain, ScatterGatherPass pass, uint[] map, TensorRef? data, TensorRef y, long total)
    {
        string type = RawType(y.Type);
        var sb = new StringBuilder();
        sb.AppendLine("struct Params {");
        sb.AppendLine($"  {type}* data;\n  uint* map;");
        for (int s = 0; s < pass.Buffers.Count; s++)
        {
            sb.AppendLine($"  {type}* s{s};");
        }

        sb.AppendLine($"  {type}* out;\n  uint n;\n}};");
        sb.AppendLine("[shader(\"compute\")] [numthreads(256, 1, 1)]");
        sb.AppendLine("void main(uint3 dtid : SV_DispatchThreadID, uniform Params p) {");
        sb.AppendLine("  uint e = dtid.x; if (e >= p.n) return;");
        sb.AppendLine("  uint v = p.map[e];");
        sb.AppendLine(data is null
            ? $"  if (v == {ScatterKeep}u) return;"
            : $"  if (v == {ScatterKeep}u) {{ p.out[e] = p.data[e]; return; }}");
        sb.AppendLine($"  uint slot = v >> {ScatterSlotShift}u; uint off = v & {ScatterMaxOffset}u;");
        for (int s = 0; s < pass.Buffers.Count; s++)
        {
            sb.AppendLine($"  {(s == 0 ? "" : "else ")}if (slot == {s}u) p.out[e] = p.s{s}[off];");
        }

        sb.AppendLine("}");
        var kernel = _compiler.Compile("scatter_gather.slang", sb.ToString(), "main", _opt.Target, Macros(), _opt.Cache);

        var mapBytes = MemoryMarshal.AsBytes(map.AsSpan()).ToArray();
        var mapBuffer = NewBuffer(BufferKind.Weight, $"const:scatter-map {chain[^1].Output.Name}", mapBytes.Length, init: mapBytes);
        var mapRef = new TensorRef(mapBuffer, 0, DataType.U32, [total], [1], null);
        var args = new List<KernelArg> { Pointer(data ?? y), Pointer(mapRef) };
        for (int s = 0; s < pass.Buffers.Count; s++)
        {
            args.Add(new PointerArg(pass.Buffers[s], pass.BaseOffsets[s]));
        }

        args.Add(Pointer(y));
        args.Add(new UIntArg((uint)total));
        var reads = new List<TensorRef> { mapRef };
        if (data is not null)
        {
            reads.Add(data);
        }

        reads.AddRange(pass.Updates);
        Emit("scatter_gather", chain, [Always(kernel, _ => new Grid(Grid.Div(total, 256)))], args, reads, [y]);
    }

    /// <summary>ScatterND without reductions: one thread per update element (duplicate indices write in unspecified order, as ONNX allows).</summary>
    private void LowerScatterNDParallel(Node n)
    {
        var data = Ref(n.Inputs[0]);
        var ix = Ref(n.Inputs[1]);
        var upd = Ref(n.Inputs[2]);
        var y = Produce(n.Output);
        EmitCopy(data, y, "scatter-copy", [n]);
        int b = (int)n.Attributes.Get("batch_dims", 0L);
        int q = (int)ix.Shape[^1];
        int iq = ix.Rank - 1;
        int tailRank = data.Rank - b - q;
        var iter = ix.Shape.Take(iq).Concat(data.Shape.Skip(b + q)).ToArray();
        var indexCoords = Enumerable.Range(0, iq).Select(Dn).ToArray();
        var sb = new StringBuilder();
        var ycoords = new string[data.Rank];
        for (int j = 0; j < data.Rank; j++)
        {
            if (j < b)
            {
                ycoords[j] = Dn(j);
            }
            else if (j < b + q)
            {
                var ic = indexCoords.Append((j - b).ToString()).ToArray();
                sb.AppendLine($"  int i{j} = int(p.in0[{Lin(ix, ic)}]); if (i{j} < 0) i{j} += {data.Shape[j]}; if (i{j} < 0 || i{j} >= {data.Shape[j]}) return;");
                ycoords[j] = $"uint(i{j})";
            }
            else
            {
                ycoords[j] = Dn(iq + (j - b - q));
            }
        }

        var ucoords = indexCoords.Concat(Enumerable.Range(0, tailRank).Select(t => Dn(iq + t))).ToArray();
        sb.AppendLine($"  p.out0[{Lin(y, ycoords)}] = p.in1[{Lin(upd, ucoords)}];");
        EmitGen("scatter_nd", [n], iter, ix.Symbol, [ix, upd], [y], sb.ToString(), outsAreReads: true);
    }
}
