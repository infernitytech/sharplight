using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Runtime;

namespace Sharplight.Families.Tests;

/// <summary>
/// An MTP draft file (only the NextN block, as llama.cpp's converter writes it with <c>--mtp</c>, next to the model) loaded as a model
/// of its own is refused with the draft-file error, whichever family it belongs to, instead of failing on a missing trunk weight.
/// </summary>
public class MtpDraftFileTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    static MtpDraftFileTests() => GgufImporter.Register();

    [Theory]
    // tools/mk_tiny_nemotron_h_mtp_hf.py, converted with convert_hf_to_gguf.py --mtp (llama.cpp 8ca5a80).
    [InlineData("gguf/tiny-random-nemotron-h-latent/mtp-tiny-random-nemotron-h-mtp-f32.gguf")]
    public void DraftFile_AsModel_IsRefusedAsADraft(string model)
    {
        string path = Path.Combine(Models, model);
        Assert.SkipUnless(File.Exists(path), $"{path} not present");
        var cpu = new CpuExecutionProvider();
        using var device = cpu.OpenDevice(cpu.GetDevices()[0]);

        var error = Assert.Throws<NotSupportedException>(() =>
            LlmModel.Load(path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 64, MaxBatchTokens = 8 }).Dispose());

        Assert.Contains("MTP draft file", error.Message, StringComparison.Ordinal);
    }
}
