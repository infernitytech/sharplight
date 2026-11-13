using System.Text.RegularExpressions;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Full-match test of each string against a regular expression (see <c>StandardOps.RegexMatch</c>).</summary>
internal sealed unsafe class RegexMatchOp(TRef x, TRef y, string pattern) : CpuOp
{
    private readonly Regex _regex = new($"^(?:{pattern})\\z", RegexOptions.CultureInvariant);

    public override string Name => "RegexMatch";

    public override void Run(RunParameters p)
    {
        long n = x.Elements(p);
        for (long i = 0; i < n; i++)
        {
            ((float*)y.P)[i] = _regex.IsMatch(p.Strings[((int*)x.P)[i]]) ? 1f : 0f;
        }
    }
}
