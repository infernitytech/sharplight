using System.Runtime.InteropServices;

namespace Sharplight.Metal.Interop;

/// <summary>Metal's <c>MTLSize</c>: three <c>NSUInteger</c> extents.</summary>
/// <param name="Width">Extent in x.</param>
/// <param name="Height">Extent in y.</param>
/// <param name="Depth">Extent in z.</param>
[StructLayout(LayoutKind.Sequential)]
internal readonly record struct MTLSize(nuint Width, nuint Height, nuint Depth);
