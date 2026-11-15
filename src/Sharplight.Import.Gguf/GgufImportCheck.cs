namespace Sharplight.Import.Gguf;

/// <summary>
/// The result of checking a GGUF file against its importer mapping from the header alone (<see cref="GgufSupport.Validate"/>): the
/// reasons the import would fail, and whether every step could be checked.
/// </summary>
/// <param name="Problems">Why importing fails (the importer's messages, without a file path); empty when it maps.</param>
/// <param name="IsComplete">
/// Every step of the mapping ran on the header; false when a step needs tensor data or the architecture has no mapping, so an empty
/// <paramref name="Problems"/> does not prove that the import succeeds.
/// </param>
/// <param name="IsDraftModel">
/// The file holds only the draft layers of another model (e.g. the multi-token-prediction layers of a split <c>MTP/</c> file) or is a drafter of
/// its own architecture (EAGLE-3, DFlash, Gemma 4 assistants), so it is loaded as that model's draft source, not on its own;
/// <paramref name="Problems"/> then says so (and names what keeps a drafter from loading, if anything).
/// </param>
public sealed record GgufImportCheck(IReadOnlyList<string> Problems, bool IsComplete, bool IsDraftModel = false)
{
    /// <summary>No problem was found and every step was checked.</summary>
    public bool IsAccepted => Problems.Count == 0 && IsComplete;
}
