namespace Sharplight.Studio.Tests;

/// <summary>Finished feed installs collapse into one dismissible summary; running and failed ones keep their cards.</summary>
public sealed class FeedInstallSummaryTests
{
    [Fact]
    public async Task FinishedInstalls_CollapseIntoASummary_ThatCanBeDismissed()
    {
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        var catalog = rig.Catalog;
        Assert.False(catalog.HasFinishedInstalls);

        var first = catalog.Install(catalog.Entries[0]);
        await first.Completion;

        Assert.True(first.IsDone);
        Assert.False(first.ShowCard);
        Assert.True(catalog.HasFinishedInstalls);
        Assert.Equal(1, catalog.FinishedInstalls);
        Assert.Equal($"Installed {first.Name}", catalog.FinishedSummary);

        var second = catalog.Install(catalog.Entries[1]);
        await second.Completion;
        Assert.Equal(2, catalog.FinishedInstalls);
        Assert.Contains("Installed 2 packages", catalog.FinishedSummary, StringComparison.Ordinal);
        Assert.Contains(first.Name, catalog.FinishedSummary, StringComparison.Ordinal);

        catalog.ClearFinishedInstallsCommand.Execute(null);

        Assert.Empty(catalog.Installs);
        Assert.False(catalog.HasFinishedInstalls);
        Assert.Equal("", catalog.FinishedSummary);
    }

    [Fact]
    public async Task AFailedInstall_KeepsItsCard_AndIsNotInTheSummary()
    {
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        File.AppendAllText(rig.Feed.PathOf("packages/alpha.q8/1.0.0/manifest.json"), " ");

        var install = rig.Catalog.Install(rig.Catalog.Entries[0]);
        await install.Completion;

        Assert.True(install.HasError);
        Assert.True(install.ShowCard);
        Assert.False(rig.Catalog.HasFinishedInstalls);
        Assert.Single(rig.Catalog.Installs);
    }
}
