using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>The residency list is refreshed from runtime events that arrive from engine threads while models load and unload.</summary>
public sealed class ResidentModelsRaceTests
{
    [Fact]
    public async Task Refresh_FromTwoThreads_DoesNotEditTheCollectionWhileTheOtherEnumeratesIt()
    {
        using var services = new TestServices();
        services.Runtime.Add(new FakeModel("a"));
        var resident = new ResidentModelsViewModel(services.Services, new FakeNavigator());
        using var inside = new ManualResetEventSlim();
        using var second = new ManualResetEventSlim();
        resident.Models.CollectionChanged += (_, _) =>
        {
            inside.Set();
            second.Wait(TimeSpan.FromSeconds(5));
            Thread.Sleep(200);
        };
        services.Runtime.Add(new FakeModel("b"));

        var first = Task.Run(resident.Refresh);
        inside.Wait(TimeSpan.FromSeconds(5));
        var other = Task.Run(() =>
        {
            second.Set();
            resident.Refresh();
        });

        await Task.WhenAll(first, other);

        resident.Refresh();
        Assert.Equal(["a", "b"], resident.Models.Select(model => model.Id));
    }

    [Fact]
    public void Refresh_RequestedFromItsOwnNotification_RunsAfterTheCurrentOne()
    {
        using var services = new TestServices();
        services.Runtime.Add(new FakeModel("a"));
        var resident = new ResidentModelsViewModel(services.Services, new FakeNavigator());
        int nested = 0;
        resident.Models.CollectionChanged += (_, _) =>
        {
            if (nested++ == 0)
            {
                resident.Refresh();
            }
        };

        services.Runtime.Add(new FakeModel("b"));
        resident.Refresh();

        Assert.Equal(["a", "b"], resident.Models.Select(model => model.Id));
    }

    [Fact]
    public async Task LoadUnloadAndRefresh_Concurrently_NeverThrowAndEndConsistent()
    {
        using var services = new TestServices();
        var resident = new ResidentModelsViewModel(services.Services, new FakeNavigator());
        using var stop = new CancellationTokenSource();
        var refreshers = Enumerable.Range(0, 4).Select(_ => Task.Run(() =>
        {
            while (!stop.IsCancellationRequested)
            {
                resident.Refresh();
            }
        })).ToList();
        var mutator = Task.Run(() =>
        {
            for (int i = 0; i < 400; i++)
            {
                services.Runtime.Add(new FakeModel($"m{i % 5}-{i}"));
                if (i % 3 == 0)
                {
                    services.Runtime.UnloadAsync($"m{(i - 1) % 5}-{i - 1}").Wait();
                }
            }
        });

        await mutator;
        await stop.CancelAsync();
        await Task.WhenAll(refreshers);

        resident.Refresh();
        Assert.Equal(services.Runtime.LoadedModels.Select(model => model.Id), resident.Models.Select(model => model.Id));
    }
}
