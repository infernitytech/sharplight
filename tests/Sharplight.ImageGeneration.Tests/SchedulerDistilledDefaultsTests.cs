using Sharplight.Import.Diffusers;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// A diffusers folder of SD-Turbo / SDXL-Turbo samples unguided in few steps by default (2026-11-18 hot-model sweep: SD-Turbo ran with
/// guidance 7.5 and came out oversaturated).
/// </summary>
public sealed class SchedulerDistilledDefaultsTests
{
    private const string Trailing = """{ "_class_name": "EulerDiscreteScheduler", "timestep_spacing": "trailing", "steps_offset": 1 }""";
    private const string Leading = """{ "_class_name": "EulerDiscreteScheduler", "timestep_spacing": "leading", "steps_offset": 1 }""";

    [Fact]
    public void TurboFoldersRunUnguided()
    {
        var settings = SchedulerSettings.FromDiffusers(Trailing).WithDistilledDefaults("sd-turbo", null);
        Assert.Equal(0f, settings.DefaultGuidance);
        Assert.Equal(4, settings.DefaultSteps);
        Assert.Equal(0f, SchedulerSettings.FromDiffusers(Trailing).WithDistilledDefaults("snapshot", "/models/sdxl-turbo/").DefaultGuidance);
    }

    [Fact]
    public void OtherFoldersKeepTheGenericDefaults()
    {
        Assert.Null(SchedulerSettings.FromDiffusers(Trailing).WithDistilledDefaults("stable-diffusion-2-1", null).DefaultGuidance);
        Assert.Null(SchedulerSettings.FromDiffusers(Leading).WithDistilledDefaults("sd-turbo", null).DefaultGuidance);
    }
}
