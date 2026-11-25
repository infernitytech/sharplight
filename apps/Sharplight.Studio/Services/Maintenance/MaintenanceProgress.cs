namespace Sharplight.Studio.Services.Maintenance;

/// <summary>Progress of a long file operation: what is being done, and bytes done of total (total 0 when unknown).</summary>
/// <param name="Message">What is being done.</param>
/// <param name="Done">Bytes (or items) done.</param>
/// <param name="Total">Bytes (or items) in all; 0 when unknown.</param>
internal sealed record MaintenanceProgress(string Message, long Done, long Total)
{
    /// <summary>0..1, or 0 when the total is unknown.</summary>
    public double Fraction => Total <= 0 ? 0 : Math.Min(1, (double)Done / Total);
}
