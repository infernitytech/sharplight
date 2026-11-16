using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Inference;

namespace Sharplight.Studio.Tests;

/// <summary>The picker is refreshed from runtime events that can arrive from engine threads or from inside its own notifications.</summary>
public sealed class ModelPickerRaceTests
{
    [Fact]
    public void Refresh_RequestedFromItsOwnNotification_RunsAfterTheCurrentOne()
    {
        using var services = new TestServices();
        services.Runtime.Add(new FakeModel("a"));
        var picker = new ModelPickerViewModel(services.Services, "text-generation");
        int nested = 0;
        picker.Models.CollectionChanged += (_, _) =>
        {
            if (nested++ == 0)
            {
                picker.Refresh();
            }
        };

        // Adding a model during the notification used to enumerate the collection while it was being edited.
        services.Runtime.Add(new FakeModel("b"));
        picker.Refresh();

        Assert.Equal(["a", "b"], picker.Models.Select(option => option.Id));
    }

    [Fact]
    public async Task Refresh_FromTwoThreads_DoesNotEditTheCollectionWhileTheOtherEnumeratesIt()
    {
        using var services = new TestServices();
        services.Runtime.Add(new FakeModel("a"));
        var picker = new ModelPickerViewModel(services.Services, "text-generation");
        using var inside = new ManualResetEventSlim();
        using var second = new ManualResetEventSlim();
        picker.Models.CollectionChanged += (_, _) =>
        {
            inside.Set();
            second.Wait(TimeSpan.FromSeconds(5));
            Thread.Sleep(200);
        };
        services.Runtime.Add(new FakeModel("b"));

        var first = Task.Run(picker.Refresh);
        inside.Wait(TimeSpan.FromSeconds(5));
        var other = Task.Run(() =>
        {
            second.Set();
            picker.Refresh();
        });

        await Task.WhenAll(first, other);

        Assert.Equal(["a", "b"], picker.Models.Select(option => option.Id));
    }
}
