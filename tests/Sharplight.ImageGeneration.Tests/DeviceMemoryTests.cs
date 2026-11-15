namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Disposing generators, compiled graphs and models returns their memory to the shared Vulkan device (in the large-model collection,
/// which runs alone, so no other test allocates meanwhile).
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class DeviceMemoryTests
{
    [Fact]
    public async Task DisposedGeneratorAndModelReleaseDeviceMemory()
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        long before = slot!.Device.GetMemoryInfo().UsedByUs;
        long during;
        using (var model = Env.Load(Env.TinyDiffusers, slot))
        {
            using (var generator = Env.Clients.Create<IImageGenerator>(model))
            {
                var options = new ImageGenerationOptions { Steps = 2, Seed = 1 };
                await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "a red cube" }, options, TestContext.Current.CancellationToken);
                during = slot.Device.GetMemoryInfo().UsedByUs;
            }

            using var compiled = model.CompileComponent(Models.ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode);
        }

        long after = slot.Device.GetMemoryInfo().UsedByUs;
        TestContext.Current.SendDiagnosticMessage($"device memory: before {before >> 20} MiB, during {during >> 20} MiB, after {after >> 20} MiB");
        Assert.True(during > before, "generation allocated nothing on the device");
        Assert.True(after <= before, $"{(after - before) >> 10} KiB of device memory stayed allocated after disposal");
    }
}
