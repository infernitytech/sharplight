namespace Sharplight.Studio.Services.Monitoring;

/// <summary>Samples CPU, memory, device memory and engine activity.</summary>
internal interface ISystemMonitor
{
    /// <summary>Takes a sample (rates are relative to the previous call).</summary>
    public SystemSample Sample();
}
