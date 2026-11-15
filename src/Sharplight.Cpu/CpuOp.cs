using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Base class of all executable steps.</summary>
internal abstract class CpuOp
{
    public abstract string Name { get; }

    /// <summary>Shape details for the per-instance profile (<c>SHARPLIGHT_CPU_PROFILE=min</c>); empty by default.</summary>
    public virtual string Detail => string.Empty;

    public abstract void Run(RunParameters p);
    public virtual void Dispose()
    {
    }

    public override string ToString() => Name;
}
