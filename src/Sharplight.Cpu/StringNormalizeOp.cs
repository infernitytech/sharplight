using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// ONNX StringNormalizer (see <c>StandardOps.StringNormalize</c>): removes stopwords, changes case and publishes the number of kept strings.
/// </summary>
internal sealed unsafe class StringNormalizeOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly HashSet<string> _stopwords;
    private readonly string _case;
    private readonly string _countSymbol;

    public StringNormalizeOp(TRef x, TRef y, string[] stopwords, bool caseSensitive, string caseAction, string countSymbol)
    {
        _x = x;
        _y = y;
        _stopwords = new HashSet<string>(stopwords, caseSensitive ? StringComparer.Ordinal : StringComparer.OrdinalIgnoreCase);
        _case = caseAction;
        _countSymbol = countSymbol;
    }

    public override string Name => "StringNormalize";

    public override void Run(RunParameters p)
    {
        long n = _x.Elements(p);
        var input = (int*)_x.P;
        var output = (int*)_y.P;
        int kept = 0;
        for (long i = 0; i < n; i++)
        {
            string s = p.Strings[input[i]];
            if (_stopwords.Contains(s))
            {
                continue;
            }

            output[kept++] = p.Strings.Add(_case switch
            {
                "LOWER" => s.ToLowerInvariant(),
                "UPPER" => s.ToUpperInvariant(),
                _ => s,
            });
        }

        if (kept == 0)
        {
            output[kept++] = p.Strings.Add("");
        }

        p.Symbols[_countSymbol] = kept;
    }
}
