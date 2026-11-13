using Sharplight.Package;
using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>
/// ONNX models stored as packages: the compiled kernels are stored with the model and a later load of the package takes every
/// kernel from it, with results equal to the ONNX file's.
/// </summary>
public class PackageArtifactTests(ITestOutputHelper output)
{
    private const string Model = "convnext-tiny";
    private const int Batch = 1;

    [Fact]
    public void PackagedOnnxModelLoadsItsStoredKernels()
    {
        var c = RefCase.Zoo(Model);
        Assert.SkipWhen(c is null, $"zoo reference or model missing for '{Model}'");
        var device = Providers.VulkanOrSkip();
        var directory = Directory.CreateTempSubdirectory("sharplight-onnx-package-");
        try
        {
            var path = Path.Combine(directory.FullName, Model + ModelPackage.Extension);
            var options = OnnxExec.OptionsFor(c!, Batch);
            OnnxExec.Import(c!.ModelPath, options, out var definition);
            try
            {
                ModelPackage.Save(path, definition, []);
            }
            finally
            {
                foreach (var resource in definition.OwnedResources)
                {
                    resource.Dispose();
                }
            }

            var inputs = c.RuntimeInputs.Select(k => c.ReadInput(Batch, k)).ToList();
            var recording = new CompilationArtifacts();
            var fromOnnx = OnnxExec.Run(c.ModelPath, options, device, Batch, inputs, artifacts: recording);
            ModelPackage.AddArtifacts(path, recording.Export());

            var fromPackage = OnnxExec.Run(path, null, device, Batch, inputs);
            var code = fromPackage.Artifacts!.Recorded.Single(s => ArtifactDescriptor.Get(s.Descriptor, ArtifactDescriptorKeys.Tier) == ArtifactTiers.Code);
            output.WriteLine($"compile: onnx {fromOnnx.CompileMs:F0} ms, package {fromPackage.CompileMs:F0} ms; {code.FallbackHits} of {code.Entries.Count} kernels from the package");
            Assert.True(code.Entries.Count > 0);
            Assert.Equal(code.Entries.Count, code.FallbackHits);
            for (int k = 0; k < fromOnnx.Outputs.Length; k++)
            {
                Assert.Equal(fromOnnx.Outputs[k], fromPackage.Outputs[k]);
            }
        }
        finally
        {
            directory.Delete(recursive: true);
        }
    }
}
