using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

// Unique over the flattened input or the outermost slices, with its data-dependent number of values.
public sealed partial class GraphLowerer
{
    /// <summary>
    /// Unique (see <c>StandardOps.Unique</c>) in two launches with one thread per input element. The first marks the elements whose
    /// value occurs for the first time. The second compares each element with all others: its slot (the inverse index) is the number
    /// of first occurrences of smaller values (sorted) or of first occurrences before its own (unsorted); a first occurrence writes its
    /// value, index and count to its slot, and element 0 writes the number of unique values, the op's data-dependent count. NaN
    /// equals NaN and sorts first, like the CPU provider.
    /// </summary>
    private void LowerUnique(Node n)
    {
        var x = EnsureDense(Ref(n.Inputs[0]), "unique");
        var values = Produce(n.Outputs[0]);
        var firstIndices = Produce(n.Outputs[1]);
        var inverse = Produce(n.Outputs[2]);
        var counts = Produce(n.Outputs[3]);
        var count = ProduceSymbol(n.Attributes.Get<string>("count_symbol"));
        bool sorted = n.Attributes.Get("sorted", true);
        bool isFloat = IsFloatType(x.Type);
        string type = isFloat ? "float" : CType(x.Type);
        string Load(string i) => isFloat ? ToFloat(x.Type, $"p.in0[{i}]") : $"p.in0[{i}]";
        string eq = isFloat ? "(a == c || (isnan(a) && isnan(c)))" : "a == c";
        string less = isFloat ? "((isnan(c) && !isnan(a)) || c < a)" : "c < a";
        if (n.Attributes.Get("slices", false))
        {
            LowerUniqueSlices(n, x, values, firstIndices, inverse, counts, count, sorted, type, Load, eq, less);
            return;
        }

        long[] shape = x.Rank == 0 ? [1] : x.Shape;
        string? symbol = x.Rank == 0 ? null : x.Symbol;
        var first = Temp(DataType.U32, shape, symbol);
        EmitGen("unique_first", [n], shape, symbol, [x], [first], $$"""
              {{type}} a = {{Load("idx")}};
              uint f = 1u;
              for (uint j = 0; j < idx; j++) { {{type}} c = {{Load("j")}}; if ({{eq}}) { f = 0u; break; } }
              p.out0[idx] = f;
            """);
        EmitGen("unique", [n], shape, symbol, [x, first], [values, firstIndices, inverse, counts, count], $$"""
              {{type}} a = {{Load("idx")}};
              uint slot = 0u, cnt = 0u, total = 0u, firstIdx = idx;
              bool seen = false;
              for (uint j = 0; j < p.n; j++) {
                {{type}} c = {{Load("j")}};
                bool e = {{eq}};
                if (e) { cnt++; if (!seen) { seen = true; firstIdx = j; } }
                if (p.in1[j] != 0u) { total++; if ({{(sorted ? less : "!seen")}}) { slot++; } }
              }
              p.out2[idx] = int(slot);
              if (firstIdx == idx) {
                p.out0[slot] = p.in0[idx];
                p.out1[slot] = int(idx);
                p.out3[slot] = int(cnt);
              }
              if (idx == 0u) { p.out4[0] = int(total); }
            """);
    }

    /// <summary>
    /// Unique over the slices x[i, ...] along the outermost axis (attribute <c>slices</c>), one thread per slice: slices compare element by
    /// element, and sort lexicographically by their first differing element (with the element order of the flattened form).
    /// </summary>
    private void LowerUniqueSlices(Node n, TensorRef x, TensorRef values, TensorRef firstIndices, TensorRef inverse, TensorRef counts, TensorRef count,
        bool sorted, string type, Func<string, string> load, string eq, string less)
    {
        long length = x.Shape.Skip(1).Aggregate(1L, (a, b) => a * b);
        long[] shape = [x.Shape[0]];
        string L = length + "u";
        // e: slices idx and j are equal; l (sorted): slice j sorts before slice idx.
        string compare = $$"""
                bool e = true;
                bool l = false;
                for (uint t = 0; t < {{L}}; t++) {
                  {{type}} a = {{load($"idx * {L} + t")}};
                  {{type}} c = {{load($"j * {L} + t")}};
                  if (!({{eq}})) { e = false; l = {{less}}; break; }
                }
            """;
        var first = Temp(DataType.U32, shape, x.Symbol);
        EmitGen("unique_first", [n], shape, x.Symbol, [x], [first], $$"""
              uint f = 1u;
              for (uint j = 0; j < idx; j++) {
            {{compare}}
                if (e) { f = 0u; break; }
              }
              p.out0[idx] = f;
            """);
        EmitGen("unique", [n], shape, x.Symbol, [x, first], [values, firstIndices, inverse, counts, count], $$"""
              uint slot = 0u, cnt = 0u, total = 0u, firstIdx = idx;
              bool seen = false;
              for (uint j = 0; j < p.n; j++) {
            {{compare}}
                if (e) { cnt++; if (!seen) { seen = true; firstIdx = j; } }
                if (p.in1[j] != 0u) { total++; if ({{(sorted ? "l" : "!seen")}}) { slot++; } }
              }
              p.out2[idx] = int(slot);
              if (firstIdx == idx) {
                for (uint t = 0; t < {{L}}; t++) { p.out0[slot * {{L}} + t] = p.in0[idx * {{L}} + t]; }
                p.out1[slot] = int(idx);
                p.out3[slot] = int(cnt);
              }
              if (idx == 0u) { p.out4[0] = int(total); }
            """);
    }
}
