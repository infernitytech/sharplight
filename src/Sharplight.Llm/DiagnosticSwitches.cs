namespace Sharplight.Llm;

/// <summary>
/// Every <c>SHARPLIGHT_*</c> environment switch read by the Llm project, read once per process. The defaults are the shipped behavior.
/// <list type="bullet">
/// <item><c>SHARPLIGHT_DEVICE_SAMPLING=0</c>: never sample tokens on the accelerator; logits are downloaded and sampled on the CPU
/// (same as <see cref="LlmLoadOptions.DeviceSampling"/> = false).</item>
/// </list>
/// </summary>
internal static class DiagnosticSwitches
{
    /// <summary>False when device sampling is disabled for the process.</summary>
    public static bool DeviceSampling { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_DEVICE_SAMPLING") != "0";
}
