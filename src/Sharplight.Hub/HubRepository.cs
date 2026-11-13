namespace Sharplight.Hub;

/// <summary>A model repository resolved to one commit, with its file list.</summary>
/// <param name="Id">Canonical repository id (<c>owner/name</c>, with the Hub's capitalization).</param>
/// <param name="Revision">The revision that was asked for (branch, tag or commit).</param>
/// <param name="Commit">The commit sha the revision resolved to; every download is pinned to it.</param>
/// <param name="Gated">The repository requires accepting terms before files can be downloaded.</param>
/// <param name="Files">Every file of the repository at <paramref name="Commit"/>.</param>
public sealed record HubRepository(string Id, string Revision, string Commit, bool Gated, IReadOnlyList<HubFile> Files);
