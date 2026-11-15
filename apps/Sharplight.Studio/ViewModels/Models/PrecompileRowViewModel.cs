using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>Progress and result of precompiling one library model for one provider and device (or a skipped provider).</summary>
internal sealed partial class PrecompileRowViewModel : ObservableObject
{
    /// <param name="provider">The execution provider.</param>
    /// <param name="device">The device; empty for a provider-level row.</param>
    public PrecompileRowViewModel(string provider, string device)
    {
        Provider = provider;
        Device = device;
    }

    /// <summary>The execution provider.</summary>
    public string Provider { get; }

    /// <summary>The device; empty for a provider-level row.</summary>
    public string Device { get; }

    /// <summary><c>vulkan · NVIDIA ...</c>.</summary>
    public string Title => Device.Length == 0 ? Provider : $"{Provider} · {Device}";

    /// <summary>The current stage, or the result.</summary>
    [ObservableProperty]
    public partial string Status { get; private set; } = "Waiting";

    /// <summary>Working on this row right now.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIndeterminate))]
    public partial bool IsActive { get; private set; }

    /// <summary>Kernel progress 0..1 while generating portable code; 0 when unknown.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIndeterminate))]
    public partial double Progress { get; private set; }

    /// <summary>Skipped or nothing to do.</summary>
    [ObservableProperty]
    public partial bool IsSkipped { get; private set; }

    /// <summary>Active without known counts (opening, driver compilation).</summary>
    public bool IsIndeterminate => IsActive && Progress <= 0;

    /// <summary>Applies a progress report for this row.</summary>
    public void Apply(PrecompileProgress update)
    {
        switch (update.Stage)
        {
            case PrecompileStage.Opening:
                Set("Opening the device", active: true);
                break;
            case PrecompileStage.Portable:
                IsActive = true;
                Progress = update.Kernels is { Total: > 0 } kernels ? (double)kernels.Completed / kernels.Total : 0;
                Status = update.Kernels is { Total: > 0 } counts
                    ? $"Portable code: {counts.Completed:N0} / {counts.Total:N0} kernels"
                    : $"Portable code: {update.Message}";
                break;
            case PrecompileStage.DeviceSpecific:
                Set("Device code: the driver compiles the kernels", active: true);
                break;
            case PrecompileStage.Completed:
                Set(update.Message, active: false);
                break;
            case PrecompileStage.Skipped:
                Set(update.Message, active: false);
                IsSkipped = true;
                break;
            default:
                Status = update.Message;
                break;
        }
    }

    /// <summary>Shows the final result of the device.</summary>
    public void Complete(DevicePrecompileResult result)
    {
        Set(Describe(result), active: false);
        IsSkipped = result.Portable is null && result.DeviceCode is null;
    }

    /// <summary>Shows why the provider or device was skipped.</summary>
    public void Skip(string reason)
    {
        Set(reason, active: false);
        IsSkipped = true;
    }

    /// <summary><c>portable spirv 312 kernels 4.1 MB · device pipeline-binary 312 entries 9.8 MB (shared with ...)</c>.</summary>
    public static string Describe(DevicePrecompileResult result)
    {
        var parts = new List<string>();
        if (result.Portable is { } portable)
        {
            parts.Add($"Portable {portable.Kind}: {ByteFormat.Count(portable.Entries, "kernel", "kernels")}, {ByteFormat.Format(portable.Bytes)}{Shared(portable)}");
        }

        if (result.DeviceCode is { } device)
        {
            parts.Add($"device {device.Kind}: {ByteFormat.Count(device.Entries, "entry", "entries")}, {ByteFormat.Format(device.Bytes)}{Shared(device)}");
        }

        if (result.Note is { } note)
        {
            parts.Add(note);
        }

        return parts.Count == 0 ? "Nothing to precompile" : string.Join(" · ", parts);
    }

    private static string Shared(CodeSetSummary set) => set.SharedWith is { } other ? $" (shared with {other})" : "";

    private void Set(string status, bool active)
    {
        Status = status;
        IsActive = active;
        Progress = 0;
    }
}
