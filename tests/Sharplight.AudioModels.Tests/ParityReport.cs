namespace Sharplight.AudioModels.Tests;

/// <summary>Appends the measured parity numbers to the file named by <c>SHARPLIGHT_PARITY_REPORT</c> (when set), so a run's figures can be read off for documentation.</summary>
internal static class ParityReport
{
    /// <summary>Appends a line.</summary>
    /// <param name="line">The line.</param>
    public static void Write(string line)
    {
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_PARITY_REPORT") is { Length: > 0 } path)
        {
            File.AppendAllText(path, line + Environment.NewLine);
        }
    }
}
