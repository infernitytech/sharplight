using Sharplight.Storage;

namespace Sharplight.Studio.Services.Settings;

/// <summary>A change of the storage location asked for on the settings page.</summary>
/// <param name="Mode">The new mode.</param>
/// <param name="CustomRoot">The folder of <see cref="StorageMode.Custom"/>; ignored otherwise.</param>
/// <param name="Transfer">What happens to the existing files.</param>
/// <param name="KeepModels">Leave the model store where it is (it stays in use as a custom model folder).</param>
internal sealed record StorageChangeRequest(StorageMode Mode, string? CustomRoot, StorageTransferKind Transfer, bool KeepModels);
