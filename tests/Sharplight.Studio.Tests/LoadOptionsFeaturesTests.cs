using Avalonia.Controls;
using Avalonia.Logging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Sharplight.Hub;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The load panel's engine options (experts on disk, FP8 and FP4 tensor-core modes, the step budget of several slots): each is enabled only
/// where it applies to the device and the model, says why otherwise, and reaches the load request; the view binds them. Also the task pages
/// every runnable kind opens.
/// </summary>
public sealed class LoadOptionsFeaturesTests
{
    private const long GiB = 1L << 30;

    /// <summary>A GPU with the block-scaled FP8 and FP4 MMAs.</summary>
    private static readonly ComputeDeviceOption s_blockScaledGpu = new("cuda:0", "Test Blackwell", "Cuda", true, 32 * GiB,
        ["fp16", DeviceFeatures.BlockScaledFp8, DeviceFeatures.BlockScaledFp4]);

    static LoadOptionsFeaturesTests() => LoadOptionsViewModel.EstimateDelay = TimeSpan.FromMilliseconds(1);

    private static LibraryEntry Entry(string kind = Sharplight.Models.ModelKinds.TextGeneration) => new()
    {
        Id = "llm",
        DisplayName = "llm",
        Path = "/models/llm/model.slm",
        SizeBytes = 4 * GiB,
        Kind = kind,
        Manifest = new ModelManifest { Repository = "org/llm", Tag = "", Alias = "llm", Commit = "abc" },
    };

    private static async Task<LoadOptionsViewModel> OpenAsync(TestServices services, LibraryEntry? entry = null)
    {
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(entry ?? Entry()), services.Services, _ => { }, () => { });
        await panel.WeightProbe.WaitAsync(TestContext.Current.CancellationToken);
        await panel.Estimate.WaitAsync(TestContext.Current.CancellationToken);
        return panel;
    }

    [Fact]
    public async Task LowPrecisionModes_AreOff_OnADeviceWithoutTheTensorCoreFeatures()
    {
        using var services = new TestServices();
        services.Runtime.WeightFormats = new ModelWeightFormats(BlockScaledFp8: true, Nvfp4: true);
        var panel = await OpenAsync(services);

        Assert.Equal(FakeRuntime.Gpu, panel.SelectedDevice);
        Assert.False(panel.CanUseBlockScaledFp8);
        Assert.False(panel.CanUseFp4Activations);
        Assert.StartsWith("Not available on this device", panel.BlockScaledFp8Hint, StringComparison.Ordinal);
        Assert.StartsWith("Not available on this device", panel.Fp4ActivationsHint, StringComparison.Ordinal);

        // A choice that does not apply is sent as the engine's default.
        panel.BlockScaledFp8 = false;
        panel.Fp4Activations = false;
        var settings = panel.ToRequest().Settings;
        Assert.True(settings.BlockScaledFp8);
        Assert.True(settings.Fp4Activations);
    }

    [Fact]
    public async Task LowPrecisionModes_FollowTheModelsWeights_AndReachTheRequest()
    {
        using var services = new TestServices();
        services.Runtime.Devices = [s_blockScaledGpu, FakeRuntime.Cpu];
        services.Runtime.WeightFormats = new ModelWeightFormats(BlockScaledFp8: true, Nvfp4: false);
        var panel = await OpenAsync(services);

        Assert.NotNull(panel.WeightFormats);
        Assert.True(panel.CanUseBlockScaledFp8);
        Assert.True(panel.BlockScaledFp8);
        Assert.DoesNotContain("Not available", panel.BlockScaledFp8Hint, StringComparison.Ordinal);
        Assert.False(panel.CanUseFp4Activations);
        Assert.StartsWith("Does not apply: this model has no NVFP4 weights", panel.Fp4ActivationsHint, StringComparison.Ordinal);

        panel.BlockScaledFp8 = false;
        Assert.False(panel.ToRequest().Settings.BlockScaledFp8);

        // The CPU has neither MMA.
        panel.SelectedDevice = FakeRuntime.Cpu;
        Assert.False(panel.CanUseBlockScaledFp8);
        Assert.True(panel.ToRequest().Settings.BlockScaledFp8);
    }

    [Fact]
    public async Task Fp4Activations_ApplyToNvfp4Models_UnlessTurnedOffInTheSettings()
    {
        using var services = new TestServices();
        services.Runtime.Devices = [s_blockScaledGpu, FakeRuntime.Cpu];
        services.Runtime.WeightFormats = new ModelWeightFormats(BlockScaledFp8: false, Nvfp4: true);
        var panel = await OpenAsync(services);
        Assert.True(panel.CanUseFp4Activations);
        Assert.StartsWith("Does not apply: this model has no block-scaled FP8 weights", panel.BlockScaledFp8Hint, StringComparison.Ordinal);
        panel.Fp4Activations = false;
        Assert.False(panel.ToRequest().Settings.Fp4Activations);

        using var disabled = new TestServices(new AppSettings { DisabledFeatures = [DeviceFeatures.BlockScaledFp4] });
        disabled.Runtime.Devices = [s_blockScaledGpu, FakeRuntime.Cpu];
        disabled.Runtime.WeightFormats = new ModelWeightFormats(BlockScaledFp8: false, Nvfp4: true);
        var off = await OpenAsync(disabled);
        Assert.False(off.CanUseFp4Activations);
        Assert.StartsWith("Turned off for every model in Settings", off.Fp4ActivationsHint, StringComparison.Ordinal);
    }

    [Fact]
    public async Task StepBudget_AppliesWithSeveralSlots_Only()
    {
        using var services = new TestServices();
        var panel = await OpenAsync(services);
        Assert.Equal(1, panel.Slots);
        Assert.False(panel.CanTuneSteps);
        Assert.Contains("two or more parallel slots", panel.StepTuningHint, StringComparison.Ordinal);
        panel.StepFloor = panel.StepFloorChoices.Single(choice => choice.Value == StepFloorPolicy.Full);
        panel.StepRowsPerSequence = 96;
        Assert.Null(panel.ToRequest().Settings.StepFloor);
        Assert.Null(panel.ToRequest().Settings.StepRowsPerSequence);

        panel.Slots = 8;
        Assert.True(panel.CanTuneSteps);
        var settings = panel.ToRequest().Settings;
        Assert.Equal(StepFloorPolicy.Full, settings.StepFloor);
        Assert.Equal(96, settings.StepRowsPerSequence);

        // 0 rows and the default choice keep the engine's defaults.
        panel.StepRowsPerSequence = 0;
        panel.StepFloor = panel.StepFloorChoices[0];
        settings = panel.ToRequest().Settings;
        Assert.Null(settings.StepFloor);
        Assert.Null(settings.StepRowsPerSequence);
    }

    [Fact]
    public async Task ExpertsOnDisk_NeedsCpuOffload_OnAGpu()
    {
        using var services = new TestServices();
        var panel = await OpenAsync(services);
        Assert.True(panel.CanOffload);
        panel.ExpertsOnDisk = true;
        Assert.False(panel.ToRequest().Settings.ExpertsOnDisk);
        panel.CpuOffload = true;
        var settings = panel.ToRequest().Settings;
        Assert.True(settings.CpuOffload);
        Assert.True(settings.ExpertsOnDisk);

        panel.SelectedDevice = FakeRuntime.Cpu;
        Assert.False(panel.CanOffload);
        Assert.False(panel.ToRequest().Settings.ExpertsOnDisk);
    }

    [Fact]
    public async Task TaskModels_DoNotProbeWeights_OrShowTheLanguageModelOptions()
    {
        using var services = new TestServices();
        var panel = await OpenAsync(services, Entry(Sharplight.Models.ModelKinds.VideoEnhancement));
        Assert.False(panel.IsLanguageModel);
        Assert.Null(panel.WeightFormats);
    }

    [Fact]
    public Task View_BindsTheOptions_AndEnablesThemByApplicability() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        services.Runtime.Devices = [s_blockScaledGpu, FakeRuntime.Cpu];
        services.Runtime.WeightFormats = new ModelWeightFormats(BlockScaledFp8: true, Nvfp4: false);
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry()), services.Services, _ => { }, () => { });
        var window = new Window { Content = new LoadOptionsView { DataContext = panel }, Width = 520, Height = 1400 };
        window.Show();
        try
        {
            await panel.WeightProbe;
            await HeadlessApp.RenderAsync(window);
            var view = window.GetVisualDescendants().OfType<LoadOptionsView>().Single();
            var fp8 = view.FindControl<ToggleSwitch>("BlockScaledFp8Toggle")!;
            var fp4 = view.FindControl<ToggleSwitch>("Fp4ActivationsToggle")!;
            var floor = view.FindControl<ComboBox>("StepFloorPicker")!;
            var rows = view.FindControl<NumericUpDown>("StepRowsInput")!;
            Assert.True(fp8.IsEffectivelyEnabled);
            Assert.True(fp8.IsChecked);
            Assert.False(fp4.IsEffectivelyEnabled);
            Assert.Equal(panel.Fp4ActivationsHint, ToolTip.GetTip(fp4));
            Assert.False(floor.IsEffectivelyEnabled);
            Assert.False(rows.IsEffectivelyEnabled);

            fp8.IsChecked = false;
            Assert.False(panel.BlockScaledFp8);
            panel.Slots = 4;
            await HeadlessApp.RenderAsync(window);
            Assert.True(floor.IsEffectivelyEnabled);
            Assert.True(rows.IsEffectivelyEnabled);
            floor.SelectedIndex = 3;
            Assert.Equal(StepFloorPolicy.Full, panel.StepFloor.Value);

            panel.SelectedDevice = FakeRuntime.Cpu;
            await HeadlessApp.RenderAsync(window);
            Assert.False(fp8.IsEffectivelyEnabled);
        }
        finally
        {
            panel.CloseCommand.Execute(null);
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    [Fact]
    public void EveryRunnableKind_OpensATaskPage_ThatConstructsWithItsViewModels()
    {
        using var services = new TestServices();
        var window = new MainWindowViewModel(services.Services);
        var pages = window.Pages.OfType<InferencePageViewModel>().ToList();
        Assert.Contains(pages, page => page is ViewModels.Video.VideoEnhancementPageViewModel);
        foreach (var page in pages)
        {
            Assert.False(string.IsNullOrEmpty(page.Title));
            Assert.False(string.IsNullOrEmpty(page.KindNoun));
            Assert.NotNull(page.Parameters);
            Assert.NotNull(page.Queue);
            Assert.False(page.HasModel);
        }

        foreach (var kind in ModelKindCatalog.All.Where(kind => kind.Runnable))
        {
            Assert.True(window.OpenKind(kind.Id, null), kind.Id);
        }
    }
}
