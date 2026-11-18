namespace Sharplight.Studio.ViewModels.Search;

/// <summary>The options of the Search page's runs.</summary>
/// <param name="TopK">How many passages a search lists.</param>
/// <param name="ChunkCharacters">The length of the passages documents are cut into, in characters.</param>
/// <param name="OverlapCharacters">How far consecutive passages overlap, in characters.</param>
internal sealed record SearchOptions(int TopK, int ChunkCharacters, int OverlapCharacters);
