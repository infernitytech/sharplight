namespace Sharplight.Studio.ViewModels.Search;

/// <summary>What an indexing run did.</summary>
/// <param name="Index">The index after the run.</param>
/// <param name="Added">Documents that were not in the index.</param>
/// <param name="Changed">Documents whose content changed.</param>
/// <param name="Removed">Documents that are gone from the folder.</param>
/// <param name="Unchanged">Documents kept as they were.</param>
/// <param name="Failed">Documents that could not be read (too large, damaged); they are retried when they change.</param>
internal sealed record IndexUpdate(SearchIndex Index, int Added, int Changed, int Removed, int Unchanged, int Failed);
