using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Base class of all executable steps.</summary>
internal abstract class CpuOp
{
    public abstract string Name { get; }
    public abstract void Run(RunParameters p);
    public virtual void Dispose() { }
    public override string ToString() => Name;
}
