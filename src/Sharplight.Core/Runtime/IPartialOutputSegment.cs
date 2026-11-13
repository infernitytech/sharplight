using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>
/// Optional capability of an <see cref="ICompiledSegment"/>: reading a byte range of an output, so a large output (the logits
/// of a big batch) can be downloaded by several threads in parallel. <see cref="CompiledModel.GetOutput(string, long, Span{byte})"/>
/// falls back to a whole read for segments without it.
/// </summary>
public interface IPartialOutputSegment
{
    /// <summary>
    /// Copies <c>destination.Length</c> bytes of an output (dense logical layout) starting at <paramref name="byteOffset"/>, after
    /// <see cref="ICompiledSegment.Execute(RunParameters)"/>. Safe to call from several threads at once between executions.
    /// </summary>
    public void GetOutput(Value value, long byteOffset, Span<byte> destination);
}
