using System.Runtime.InteropServices;

namespace Sharplight.Cpu.Tests;

internal static class SpanExtensions
{
    extension(Span<int> span)
    {
        /// <summary>Reinterprets the i32 bits as f32 without copying.</summary>
        public Span<float> Cast() => MemoryMarshal.Cast<int, float>(span);
    }
}
