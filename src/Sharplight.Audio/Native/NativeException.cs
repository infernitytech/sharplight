namespace Sharplight.Audio.Native;

/// <summary>Maps native result codes to .NET exceptions.</summary>
internal static class NativeException
{
    /// <summary>Throws the exception matching a failed result; does nothing for success.</summary>
    /// <param name="result">The native result.</param>
    /// <param name="operation">What was attempted, for the message.</param>
    public static void ThrowIfFailed(int result, string operation)
    {
        if (result >= 0)
        {
            return;
        }

        string message = $"{operation} failed: {Describe(result)}.";
        throw (NativeResult)result switch
        {
            NativeResult.InvalidArgument => new ArgumentException(message),
            NativeResult.OutOfMemory => new OutOfMemoryException(message),
            NativeResult.Io => new IOException(message),
            NativeResult.Unsupported => new InvalidDataException(message),
            NativeResult.Decode => new InvalidDataException(message),
            _ => new InvalidOperationException(message),
        };
    }

    private static string Describe(int result) => (NativeResult)result switch
    {
        NativeResult.InvalidArgument => "invalid argument",
        NativeResult.OutOfMemory => "out of memory",
        NativeResult.NoBackend => "no audio backend or device available",
        NativeResult.Io => "I/O error",
        NativeResult.Unsupported => "unsupported or unrecognized audio format",
        NativeResult.Decode => "corrupt or truncated audio data",
        NativeResult.Device => "the audio device could not be opened or started",
        NativeResult.InvalidState => "invalid state",
        NativeResult.BufferTooSmall => "buffer too small",
        _ => $"error {result}",
    };
}
