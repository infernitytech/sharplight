using System.Text;
using Sharplight.Gguf;
using Sharplight.Hub;
using Sharplight.Models;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Before downloading: whether a model is likely to import and run, from the metadata read with range requests (GGUF header and tensor
/// table, config.json, ONNX opsets) or the search summary; the badges, the box above the download button and the library's errors.
/// </summary>
public sealed class CompatibilityTests
{
    private static readonly string s_models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    private static string Gguf(string name) => Path.Combine(s_models, "gguf", name);

    private static string Hf(string name) => Path.Combine(s_models, "hf", name, "config.json");

    private static async Task<(ModelProbe Probe, LocalRanges Ranges)> ProbeGgufAsync(string path, bool readWeightFormats = true)
    {
        var ranges = new LocalRanges(path);
        var probe = await VariantProbe.ProbeGgufAsync(ranges.ReadAsync, readWeightFormats, TestContext.Current.CancellationToken);
        return (probe, ranges);
    }

    [Fact]
    public async Task Gguf_UnsupportedArchitecture()
    {
        string path = Gguf("tiny-random-hrm-text-f32.gguf");
        Assert.SkipUnless(File.Exists(path), "test GGUF not found");
        var (probe, _) = await ProbeGgufAsync(path);
        Assert.Equal(CompatibilityState.UnsupportedArchitecture, probe.Compatibility.State);
        Assert.Equal("hrm_text", probe.Compatibility.Subject);
        Assert.Equal("Unsupported architecture hrm_text", probe.Compatibility.Text);
        Assert.True(probe.Compatibility.IsUnsupported);
        Assert.Equal(CompatibilityState.UnsupportedArchitecture, probe.ArchitectureCompatibility.State);
    }

    [Fact]
    public async Task Gguf_Projector_IsNotAModel()
    {
        string path = Gguf("mmproj-tiny-random-glm4v-f32.gguf");
        Assert.SkipUnless(File.Exists(path), "test GGUF not found");
        var (probe, _) = await ProbeGgufAsync(path);
        Assert.Equal(CompatibilityState.Projector, probe.Compatibility.State);
        Assert.Contains("not a model", probe.Compatibility.Reason, StringComparison.Ordinal);
        Assert.False(probe.Compatibility.IsUnsupported);
    }

    [Fact]
    public async Task Gguf_PartialArchitecture_FromTheMapping()
    {
        string path = Gguf("tiny-random-qwen3.8-flash-next-sparse-f32.gguf");
        Assert.SkipUnless(File.Exists(path), "test GGUF not found");
        var (probe, _) = await ProbeGgufAsync(path);
        Assert.Equal(CompatibilityState.Partial, probe.Compatibility.State);
        Assert.Equal("Partially supported", probe.Compatibility.Text);
        Assert.Contains("dense regime", probe.Compatibility.Reason, StringComparison.Ordinal);
        Assert.True(probe.Compatibility.WeightsChecked);
    }

    [Fact]
    public async Task Gguf_QuantizedModel_ReadsTheTensorTableFromAPrefix()
    {
        string path = Gguf("quant-test/qwen3-0.6b-IQ2_S.gguf");
        Assert.SkipUnless(File.Exists(path), "test GGUF not found");
        var (probe, ranges) = await ProbeGgufAsync(path);
        Assert.Equal(CompatibilityState.Supported, probe.Compatibility.State);
        Assert.True(probe.Compatibility.WeightsChecked);
        Assert.Contains("IQ2_S", probe.Compatibility.Reason, StringComparison.Ordinal);
        Assert.NotNull(probe.Shape);
        Assert.Equal(28, probe.Shape.Layers);
        Assert.True(ranges.BytesRead < new FileInfo(path).Length / 4, $"read {ranges.BytesRead} bytes");

        // Without the weight formats only the first megabyte is read.
        var (shapeOnly, shapeRanges) = await ProbeGgufAsync(path, readWeightFormats: false);
        Assert.False(shapeOnly.Compatibility.WeightsChecked);
        Assert.Equal(CompatibilityState.Supported, shapeOnly.Compatibility.State);
        Assert.Equal(1, shapeRanges.Reads);
    }

    [Fact]
    public async Task Gguf_TypesWithoutKernels_AreUnsupportedWeightFormats()
    {
        using var dir = new TempDir();
        string q8k = dir.File("q8k.gguf");
        File.WriteAllBytes(q8k, TinyGguf("llama", ("token_embd.weight", 0), ("blk.0.attn_q.weight", 15), ("blk.0.attn_k.weight", 15)));
        var (probe, _) = await ProbeGgufAsync(q8k);
        Assert.Equal(CompatibilityState.UnsupportedWeightFormat, probe.Compatibility.State);
        Assert.Equal("Unsupported weight format Q8_K", probe.Compatibility.Text);
        Assert.Contains("2 tensors are stored as Q8_K", probe.Compatibility.Reason, StringComparison.Ordinal);
        // The architecture alone is fine: other quantizations of the repository share that verdict until they are read.
        Assert.Equal(CompatibilityState.Supported, probe.ArchitectureCompatibility.State);
        Assert.False(probe.ArchitectureCompatibility.WeightsChecked);

        string unknown = dir.File("unknown.gguf");
        File.WriteAllBytes(unknown, TinyGguf("qwen3", ("output.weight", 99)));
        (probe, _) = await ProbeGgufAsync(unknown);
        Assert.Equal("Unsupported weight format ggml type id 99", probe.Compatibility.Text);
    }

    [Fact]
    public void Config_RealCheckpoints()
    {
        string[] names = ["tiny-random-hrm-text", "tiny-random-kimi-k3", "tiny-random-gpt-oss-mxfp4", "tiny-random-qwen3.8-flash-next", "Qwen3-0.6B"];
        Assert.SkipUnless(names.All(name => File.Exists(Hf(name))), "test configs not found");
        ModelCompatibility Check(string name) => CompatibilityChecker.FromConfig(File.ReadAllText(Hf(name)));

        Assert.Equal("Unsupported architecture hrm_text", Check("tiny-random-hrm-text").Text);
        Assert.Equal(CompatibilityState.UnsupportedArchitecture, Check("tiny-random-kimi-k3").State);
        var mxfp4 = Check("tiny-random-gpt-oss-mxfp4");
        Assert.Equal(CompatibilityState.Supported, mxfp4.State);
        Assert.Contains("mxfp4", mxfp4.Reason, StringComparison.Ordinal);
        Assert.Equal(CompatibilityState.Partial, Check("tiny-random-qwen3.8-flash-next").State);
        var plain = Check("Qwen3-0.6B");
        Assert.Equal(CompatibilityState.Supported, plain.State);
        Assert.True(plain.WeightsChecked);
    }

    [Fact]
    public void Config_Quantization()
    {
        var gptq = CompatibilityChecker.FromConfig("""{"model_type": "qwen2", "quantization_config": {"bits": 4, "quant_method": "gptq"}}""");
        Assert.Equal("Unsupported weight format gptq", gptq.Text);
        var mlx = CompatibilityChecker.FromConfig("""{"model_type": "qwen3", "quantization_config": {"group_size": 64, "bits": 4}}""");
        Assert.Equal("Unsupported weight format mlx (4-bit)", mlx.Text);
        var modelOpt = CompatibilityChecker.FromConfig("""{"model_type": "nemotron_h"}""", """{"quantization": {"quant_algo": "NVFP4"}}""");
        Assert.Equal("Unsupported weight format modelopt NVFP4", modelOpt.Text);
        Assert.Equal(CompatibilityState.Unknown, CompatibilityChecker.FromConfig("""{"model_type": "qwen3", "quantization_config": {"quant_method": "new-thing"}}""").State);
        Assert.Equal(CompatibilityState.Unknown, CompatibilityChecker.FromConfig("""{"hidden_size": 8}""").State);
        // An unsupported architecture is named first, whatever the weights.
        Assert.Equal("Unsupported architecture hrm_text", CompatibilityChecker.FromConfig("""{"model_type": "hrm_text", "quantization_config": {"quant_method": "awq"}}""").Text);
    }

    [Fact]
    public async Task Onnx_OpsetsAreReadBehindTheGraph()
    {
        string small = Path.Combine(s_models, "onnx/mnist-12.onnx");
        string large = Path.Combine(s_models, "onnx/sam2-tiny/vision_encoder.onnx");
        Assert.SkipUnless(File.Exists(small) && File.Exists(large), "test ONNX models not found");

        foreach (var path in new[] { small, large })
        {
            var ranges = new LocalRanges(path);
            long length = new FileInfo(path).Length;
            var verdict = await VariantProbe.ProbeOnnxAsync(ranges.ReadAsync, length, TestContext.Current.CancellationToken);
            Assert.Equal(CompatibilityState.CheckedAtImport, verdict.State);
            Assert.Contains("Opsets: ai.onnx ", verdict.Reason, StringComparison.Ordinal);
            Assert.Contains("checked when the model is imported", verdict.Reason, StringComparison.Ordinal);
            Assert.True(ranges.Reads <= 3, $"{ranges.Reads} reads");
            Assert.True(ranges.BytesRead <= 3 * 64 * 1024, $"read {ranges.BytesRead} of {length} bytes");
        }

        var mnist = await VariantProbe.ProbeOnnxAsync(new LocalRanges(small).ReadAsync, new FileInfo(small).Length, TestContext.Current.CancellationToken);
        Assert.Contains("ai.onnx 12", mnist.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void Summary_FromTheSearchResult()
    {
        HubModelSummary Summary(IReadOnlyList<ModelFormat> formats, string? arch = null, string? type = null, string? method = null, bool projectors = false) =>
            new("o/m", 1, 1, false, null, null, null, formats, [], arch, type, method, projectors);

        Assert.Equal("Unsupported architecture hrm_text", CompatibilityChecker.FromSummary(Summary([ModelFormat.Gguf], arch: "hrm_text")).Text);
        var qwen = CompatibilityChecker.FromSummary(Summary([ModelFormat.Gguf], arch: "qwen3"));
        Assert.Equal(CompatibilityState.Supported, qwen.State);
        Assert.False(qwen.WeightsChecked);
        Assert.Equal(CompatibilityState.Unknown, CompatibilityChecker.FromSummary(Summary([ModelFormat.Gguf])).State);
        Assert.Equal("Unsupported weight format gptq", CompatibilityChecker.FromSummary(Summary([ModelFormat.Safetensors], type: "qwen2", method: "gptq")).Text);
        Assert.Equal(CompatibilityState.UnsupportedWeightFormat, CompatibilityChecker.FromSummary(Summary([ModelFormat.Safetensors], type: "qwen3_5", method: "mlx")).State);
        Assert.Equal(CompatibilityState.Supported, CompatibilityChecker.FromSummary(Summary([ModelFormat.Safetensors], type: "qwen3", method: "fp8")).State);
        Assert.Equal(CompatibilityState.Projector, CompatibilityChecker.FromSummary(Summary([], arch: "clip", projectors: true)).State);
        Assert.Equal(CompatibilityState.CheckedAtImport, CompatibilityChecker.FromSummary(Summary([ModelFormat.Onnx])).State);
    }

    [Fact]
    public void Library_ImportErrorsNameTheReason()
    {
        var architecture = new UnsupportedArchitectureException("hrm_text", "model.gguf: GGUF architecture 'hrm_text' has no canonical mapping. Supported: " + new string('x', 400));
        string text = ImportFailure.Describe(new InvalidOperationException("Load failed", architecture), "Load failed");
        Assert.StartsWith("Unsupported architecture hrm_text. ", text, StringComparison.Ordinal);
        Assert.DoesNotContain("xxxx", text, StringComparison.Ordinal);

        var format = new UnsupportedWeightFormatException("gptq", "m: weights quantized with gptq are not supported.");
        Assert.StartsWith("Unsupported weight format gptq. ", ImportFailure.Describe(format, format.Message), StringComparison.Ordinal);
        Assert.Equal("Disk full", ImportFailure.Describe(new IOException("Disk full"), "Disk full"));
    }

    [Fact]
    public void Library_LoadErrorsOfLocalFiles_AreExplainedFromTheFile()
    {
        string path = Gguf("tiny-random-hrm-text-f32.gguf");
        Assert.SkipUnless(File.Exists(path), "test GGUF not found");
        var error = Assert.ThrowsAny<NotSupportedException>(() => GgufImporter.Open(path));
        Assert.StartsWith("Unsupported architecture hrm_text. ", ImportFailure.Describe(error, error.Message, path), StringComparison.Ordinal);

        // A failure that does not say why: the file is read for a verdict.
        Assert.StartsWith("Unsupported architecture hrm_text. ", ImportFailure.Describe(new InvalidOperationException("failed"), "failed", path), StringComparison.Ordinal);
        string supported = Gguf("quant-test/qwen3-0.6b-IQ2_S.gguf");
        if (File.Exists(supported))
        {
            Assert.Equal("failed", ImportFailure.Describe(new InvalidOperationException("failed"), "failed", supported));
        }
    }

    [Fact]
    public async Task Discover_BadgesRepositoriesAndQuantizations_AndReadsTheSelectedOnesWeights()
    {
        using var services = new TestServices();
        services.Hub.Results.AddRange(
        [
            new HubModelSummary("someone/HRM-Text-GGUF", 900, 1, false, "gguf", "text-generation", null, [ModelFormat.Gguf], ["Q4_K_M"], GgufArchitecture: "hrm_text"),
            new HubModelSummary("Qwen/Qwen3-8B-GGUF", 500, 1, false, "gguf", "text-generation", null, [ModelFormat.Gguf], ["Q4_K_M", "Q5_K_M", "Q8_0"], GgufArchitecture: "qwen3"),
            new HubModelSummary("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", 400, 1, false, "transformers", "text-generation", null, [ModelFormat.Safetensors], [], ModelType: "qwen2", QuantizationMethod: "gptq"),
            new HubModelSummary("Qwen/Qwen3-8B", 300, 1, false, "transformers", "text-generation", null, [ModelFormat.Safetensors], [], ModelType: "qwen3"),
        ]);
        services.Hub.Compatibility = (repository, tag) => tag == "Q8_0"
            ? new ModelCompatibility(CompatibilityState.UnsupportedWeightFormat, "Q8_K", "1 tensor is stored as Q8_K, which has no compute kernels, so loading fails.")
            : new ModelCompatibility(CompatibilityState.Supported, "qwen3", "The GGUF architecture 'qwen3' is supported.");
        var search = new HubSearchViewModel(services.Services, new DownloadsViewModel(services.Services), () => new HashSet<string>());
        search.Query = "qwen";
        await search.SearchCommand.ExecuteAsync(null);

        // Unsupported results go last; the Hub's order stays otherwise.
        Assert.Equal(["Qwen/Qwen3-8B-GGUF", "Qwen/Qwen3-8B", "someone/HRM-Text-GGUF", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"], search.Results.Select(result => result.Id));
        Assert.Equal("Supported", search.Results[0].Compatibility.Text);
        Assert.True(search.Results[2].Compatibility.IsBad);
        Assert.Equal("Unsupported architecture hrm_text", search.Results[2].Compatibility.Text);
        Assert.Equal("Unsupported weight format gptq", search.Results[3].Compatibility.Text);

        await search.FitEstimate;
        var badges = search.Variants.ToDictionary(variant => variant.Tag, variant => variant.Compatibility!);
        Assert.True(badges["Q4_K_M"].Compatibility.WeightsChecked);
        Assert.Equal("Supported", badges["Q4_K_M"].Text);
        Assert.False(badges["Q8_0"].Compatibility.WeightsChecked);
        Assert.Equal("Architecture supported", badges["Q8_0"].ShortText);
        Assert.Contains("read when this quantization is selected", badges["Q8_0"].ToolTip, StringComparison.Ordinal);
        Assert.Equal(["Q4_K_M"], services.Hub.ProbedVariants);
        Assert.True(search.SelectedCompatibility!.IsGood);

        // Selecting another quantization reads its own tensor table; the box above the download button says why, without blocking.
        search.SelectedVariant = search.Variants.Single(variant => variant.Tag == "Q8_0");
        await search.CompatibilityCheck;
        Assert.Equal(["Q4_K_M", "Q8_0"], services.Hub.ProbedVariants);
        var selected = search.SelectedCompatibility!;
        Assert.True(selected.IsBad);
        Assert.Equal("Unsupported weight format Q8_K", selected.Title);
        Assert.Equal("Unsupported Q8_K", selected.ShortText);
        Assert.EndsWith("Downloading is still possible.", selected.Detail, StringComparison.Ordinal);
        Assert.True(search.DownloadCommand.CanExecute(null));
    }

    /// <summary>A GGUF file with one string key (the architecture) and the given tensors (one 32-element row each, no data).</summary>
    private static byte[] TinyGguf(string architecture, params (string Name, int Type)[] tensors)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        void String(string value)
        {
            var bytes = Encoding.UTF8.GetBytes(value);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
        }

        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write((ulong)tensors.Length);
        writer.Write(1UL);
        String("general.architecture");
        writer.Write(8u);
        String(architecture);
        foreach (var (name, type) in tensors)
        {
            String(name);
            writer.Write(1u);
            writer.Write(256UL);
            writer.Write((uint)type);
            writer.Write(0UL);
        }

        writer.Flush();
        return stream.ToArray();
    }
}
