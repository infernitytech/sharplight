using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Core.Tests.Runtime;

/// <summary>Caps of <see cref="DeviceMemoryLimits"/>: parsing, combining with the hardware, and the out-of-memory error they lead to.</summary>
public sealed class DeviceMemoryLimitsTests
{
    [Fact]
    public void SizesParseWithBinarySuffixes()
    {
        Assert.Equal(256L << 20, DeviceMemoryLimits.ParseSize("256M"));
        Assert.Equal(256L << 20, DeviceMemoryLimits.ParseSize("256MiB"));
        Assert.Equal(3L << 29, DeviceMemoryLimits.ParseSize("1.5G"));
        Assert.Equal(64L << 10, DeviceMemoryLimits.ParseSize("64kb"));
        Assert.Equal(1024, DeviceMemoryLimits.ParseSize(" 1024 "));
        Assert.Equal(0, DeviceMemoryLimits.ParseSize("0"));
        Assert.Throws<FormatException>(() => DeviceMemoryLimits.ParseSize("lots"));
        Assert.Throws<FormatException>(() => DeviceMemoryLimits.ParseSize("-1M"));
    }

    [Fact]
    public void OptionsCapTheHardware()
    {
        var options = new DeviceOptions();
        options.MemoryLimits.DeviceMemory = 1L << 30;
        options.MemoryLimits.HostVisibleDeviceMemory = 0;
        var limits = DeviceMemoryLimits.For(options);
        Assert.Equal(1L << 30, DeviceMemoryLimits.Apply(8L << 30, limits.DeviceMemory));
        Assert.Equal(1L << 20, DeviceMemoryLimits.Apply(1L << 20, limits.DeviceMemory));
        Assert.Equal(0, DeviceMemoryLimits.Apply(256L << 20, limits.HostVisibleDeviceMemory));
        Assert.Equal(5L, DeviceMemoryLimits.Apply(5L, limits.MaxAllocation));
        Assert.Contains("host-visible device memory 0 MiB", limits.ToString());
        Assert.True(new DeviceMemoryLimits().IsEmpty);
    }

    [Fact]
    public void OutOfMemoryNamesTheAllocation()
    {
        var error = new DeviceOutOfMemoryException("GPU", "weight buffer w", 3L << 20, "Heaps: ...");
        Assert.IsAssignableFrom<OutOfMemoryException>(error);
        Assert.Contains("3.0 MiB for weight buffer w", error.Message);
        Assert.Contains("Heaps: ...", error.Message);
    }
}
