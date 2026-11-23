using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Core.Tests.Tasks;

/// <summary>
/// <see cref="TaskLoadOptions.BlockScaledFp8"/>: on by default, and turning it off disables <see cref="DeviceFeatures.BlockScaledFp8"/> for
/// the load's compilations (which is how the lowering keeps the f32-scale FP8 path) without touching <see cref="TaskLoadOptions.DisabledFeatures"/>.
/// </summary>
public class BlockScaledFp8OptionTests
{
    [Fact]
    public void OnByDefault()
    {
        Assert.True(new TaskLoadOptions().BlockScaledFp8);
        Assert.True(new LlmLoadOptions().BlockScaledFp8);
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_FP8_BLOCK_SCALED") is null)
        {
            Assert.DoesNotContain(DeviceFeatures.BlockScaledFp8, new LlmLoadOptions().CompileDisabledFeatures());
        }
    }

    [Fact]
    public void OptOutDisablesTheFeatureForCompilations()
    {
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_FP8_BLOCK_SCALED") is not null)
        {
            return;
        }

        var options = new LlmLoadOptions { BlockScaledFp8 = false };
        options.DisabledFeatures.Add(DeviceFeatures.MatrixF16);
        var disabled = options.CompileDisabledFeatures();
        Assert.Contains(DeviceFeatures.BlockScaledFp8, disabled);
        Assert.Contains(DeviceFeatures.MatrixF16, disabled);
        Assert.DoesNotContain(DeviceFeatures.BlockScaledFp8, options.DisabledFeatures);

        var effective = DeviceFeatures.Effective(new HashSet<string> { DeviceFeatures.BlockScaledFp8, DeviceFeatures.NativeGemmFp8 }, disabled);
        Assert.DoesNotContain(DeviceFeatures.BlockScaledFp8, effective);
        Assert.Contains(DeviceFeatures.NativeGemmFp8, effective);
    }
}
