using System.Globalization;
using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>Lowering of the hashed n-gram row ids (NGramHash): one thread per (token, head), constants baked into the kernel.</summary>
public sealed partial class GraphLowerer
{
    private void LowerNGramHash(Node n)
    {
        var a = n.Attributes;
        var window = EnsureDense(AsF32(Ref(n.Inputs[0]), "ngram-window"), "ngram-window");
        var y = Produce(n.Output);
        var multipliers = a.Get<long[]>("multipliers");
        long headsPerOrder = a.Get<long>("heads_per_order");
        long stop = a.Get<long>("stop_token");
        int order = multipliers.Length;

        string body = string.Join('\n',
            Table("mult", multipliers),
            Table("size", a.Get<long[]>("head_sizes")),
            Table("offs", a.Get<long[]>("head_offsets")),
            $"  int ord = int(d1 / {headsPerOrder}u) + 2;",
            "  uint64_t mixed = 0; bool cut = false;",
            "  for (int j = 0; j < ord; j++) {",
            $"    int64_t tok = int64_t(p.in0[uint64_t(d0) * {order}u + uint(j)]) - 1;",
            $"    if (j > 0) cut = cut || tok < 0 || tok == {stop};",
            $"    if ((j > 0 && cut) || tok < 0) tok = {stop};",
            "    mixed ^= uint64_t(tok) * mult[j];",
            "  }",
            "  p.out0[idx] = int(mixed % size[d1] + offs[d1]);");
        EmitGen($"ngram-hash {n.Output.Name}", [n], y.Shape, y.Symbol, [window], [y], body);

        static string Table(string name, long[] values) =>
            $"  const uint64_t {name}[{values.Length}] = {{ {string.Join(", ", values.Select(v => v.ToString(CultureInfo.InvariantCulture) + "ull"))} }};";
    }
}
