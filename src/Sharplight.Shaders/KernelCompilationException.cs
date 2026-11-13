namespace Sharplight.Shaders;

/// <summary>A Slang kernel failed to compile; the message carries the compiler diagnostics and the macros used.</summary>
/// <param name="message">Diagnostics.</param>
public sealed class KernelCompilationException(string message) : Exception(message);
