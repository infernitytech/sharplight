using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Writes the handles of constant strings (see <c>StandardOps.StringConst</c>); the strings are interned anew on every run.</summary>
internal sealed unsafe class StringConstOp(TRef y, string[] strings) : CpuOp
{
    public override string Name => "StringConst";

    public override void Run(RunParameters p)
    {
        var handles = (int*)y.P;
        for (int i = 0; i < strings.Length; i++)
        {
            handles[i] = p.Strings.Add(strings[i]);
        }
    }
}
