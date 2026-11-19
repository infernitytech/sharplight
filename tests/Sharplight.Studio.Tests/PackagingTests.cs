using System.Diagnostics;
using System.Xml.Linq;

namespace Sharplight.Studio.Tests;

/// <summary>The published Studio package leaves out the API documentation files; the NuGet packages of the libraries keep them.</summary>
public sealed class PackagingTests
{
    private static string RepositoryFile(params string[] parts)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Sharplight.slnx")))
        {
            directory = directory.Parent;
        }

        Assert.NotNull(directory);
        return Path.Combine([directory.FullName, .. parts]);
    }

    private static string? Property(string file, string name)
        => XDocument.Load(file).Descendants(name).Select(element => element.Value).LastOrDefault();

    [Fact]
    public void Studio_DoesNotPublishXmlDocumentation_ButLibrariesStillGenerateIt()
    {
        string project = RepositoryFile("apps", "Sharplight.Studio", "Sharplight.Studio.csproj");
        Assert.Equal("false", Property(project, "PublishDocumentationFile"));
        Assert.Equal("false", Property(project, "PublishReferencesDocumentationFiles"));
        Assert.Equal("true", Property(RepositoryFile("Directory.Build.props"), "GenerateDocumentationFile"));
        // The libraries that become NuGet packages do not opt out of the xml files the packages carry.
        foreach (string library in new[] { "Sharplight.Core", "Sharplight", "Sharplight.Hosting" })
        {
            Assert.NotEqual("false", Property(RepositoryFile("src", library, library + ".csproj"), "GenerateDocumentationFile"));
        }
    }

    // The CUDA execution provider and NVRTC ship for linux-x64, linux-arm64 and win-x64 and for no other runtime.
    private static readonly string[] s_cudaRids = ["linux-x64", "linux-arm64", "win-x64"];
    private static readonly string[] s_noCudaRids = ["osx-arm64", "win-arm64"];
    private static readonly string[] s_headers = ["cuda_fp16.h", "cuda_fp16.hpp", "cuda_bf16.h", "cuda_bf16.hpp", "cuda_fp8.h", "cuda_fp8.hpp"];

    private static (int Exit, string Output) Run(string file, params string[] arguments)
    {
        var info = new ProcessStartInfo(file) { RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (string argument in arguments)
        {
            info.ArgumentList.Add(argument);
        }

        using var process = Process.Start(info)!;
        string output = process.StandardOutput.ReadToEnd() + process.StandardError.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, output);
    }

    private static string[] CudaFiles(string rid) => rid switch
    {
        "win-x64" => ["Sharplight.Cuda.dll", "nvrtc64_130_0.dll", "nvrtc-builtins64_130.dll", .. s_headers],
        _ => ["Sharplight.Cuda.dll", "libnvrtc.so.13", "libnvrtc-builtins.so.13.0", .. s_headers],
    };

    private static string Publish(IEnumerable<string> files, bool documents)
    {
        string directory = Path.Combine(Path.GetTempPath(), "studio-cuda-pack-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        foreach (string file in files.Concat(documents ? ["THIRD-PARTY-NOTICES-Cuda.md", "LICENSE-NVIDIA-CUDA.txt"] : []))
        {
            File.WriteAllText(Path.Combine(directory, file), "x");
        }

        return directory;
    }

    [Fact]
    public void Studio_ReferencesTheCudaProvider_ExactlyForTheRuntimesWithAnNvidiaDriver()
    {
        string project = RepositoryFile("apps", "Sharplight.Studio", "Sharplight.Studio.csproj");
        var conditions = XDocument.Load(project).Descendants("StudioCuda").Select(element => element.Attribute("Condition")?.Value ?? "").ToList();
        foreach (string rid in s_cudaRids)
        {
            Assert.Contains(conditions, condition => condition.Contains($"'$(RuntimeIdentifier)' == '{rid}'", StringComparison.Ordinal));
        }

        foreach (string rid in s_noCudaRids)
        {
            Assert.DoesNotContain(conditions, condition => condition.Contains(rid, StringComparison.Ordinal));
        }

        // macOS and Windows on ARM never get it, also without a runtime identifier (a build on those machines).
        Assert.All(conditions.Where(condition => condition.Contains("'$(RuntimeIdentifier)' == ''", StringComparison.Ordinal)), condition =>
            Assert.True(condition.Contains("OSX", StringComparison.Ordinal) || condition.Contains("Linux", StringComparison.Ordinal) || condition.Contains("X64", StringComparison.Ordinal)));
        var references = XDocument.Load(project).Descendants("ProjectReference").Where(element => (element.Attribute("Include")?.Value ?? "").Contains("Sharplight.Cuda", StringComparison.Ordinal)).ToList();
        Assert.Equal(2, references.Count);
        Assert.All(references, reference => Assert.Equal("'$(StudioCuda)' == 'true'", reference.Parent!.Attribute("Condition")?.Value));
    }

    [Fact]
    public void StudioCuda_IsTrueForTheThreeRuntimes_AndFalseForTheOthers()
    {
        Assert.SkipWhen(OperatingSystem.IsWindows(), "evaluates the project with dotnet msbuild");
        string project = RepositoryFile("apps", "Sharplight.Studio", "Sharplight.Studio.csproj");
        string dotnet = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") ?? "dotnet";
        foreach (string rid in s_cudaRids.Concat(s_noCudaRids))
        {
            var (exit, output) = Run(dotnet, "msbuild", project, "-getProperty:StudioCuda", $"-p:RuntimeIdentifier={rid}", "-nologo");
            Assert.True(exit == 0, output);
            Assert.Equal(s_cudaRids.Contains(rid) ? "true" : "false", output.Split('\n', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)[0]);
        }
    }

    [Fact]
    public void CheckCuda_AcceptsTheCudaFilesForTheThreeRuntimes_AndRejectsAMissingOne()
    {
        Assert.SkipWhen(OperatingSystem.IsWindows(), "runs the bash script of the CI packaging");
        string script = RepositoryFile("packaging", "check-cuda.sh");
        foreach (string rid in s_cudaRids)
        {
            string complete = Publish(CudaFiles(rid), documents: true);
            var (exit, output) = Run("bash", script, complete, rid);
            Assert.True(exit == 0, rid + ": " + output);

            foreach (string missing in CudaFiles(rid).Concat(["THIRD-PARTY-NOTICES-Cuda.md", "LICENSE-NVIDIA-CUDA.txt"]))
            {
                File.Move(Path.Combine(complete, missing), Path.Combine(complete, missing + ".away"));
                var (broken, why) = Run("bash", script, complete, rid);
                Assert.True(broken != 0 && why.Contains(missing, StringComparison.Ordinal), $"{rid} without {missing}: {why}");
                File.Move(Path.Combine(complete, missing + ".away"), Path.Combine(complete, missing));
            }

            // The runtime's native folder is as good a place as the application directory.
            string native = Publish([], documents: true);
            Directory.CreateDirectory(Path.Combine(native, "runtimes", rid, "native"));
            foreach (string file in CudaFiles(rid))
            {
                File.WriteAllText(Path.Combine(native, file == "Sharplight.Cuda.dll" ? file : Path.Combine("runtimes", rid, "native", file)), "x");
            }

            Assert.Equal(0, Run("bash", script, native, rid).Exit);
        }
    }

    [Fact]
    public void CheckCuda_RejectsAnyCudaFile_InThePackagesWithoutAnNvidiaDriver()
    {
        Assert.SkipWhen(OperatingSystem.IsWindows(), "runs the bash script of the CI packaging");
        string script = RepositoryFile("packaging", "check-cuda.sh");
        foreach (string rid in s_noCudaRids)
        {
            Assert.Equal(0, Run("bash", script, Publish(["sharplight-studio"], documents: false), rid).Exit);
            foreach (string file in CudaFiles("linux-x64").Concat(CudaFiles("win-x64")).Concat(["THIRD-PARTY-NOTICES-Cuda.md", "LICENSE-NVIDIA-CUDA.txt"]).Distinct())
            {
                var (exit, output) = Run("bash", script, Publish([file], documents: false), rid);
                Assert.True(exit != 0 && output.Contains(file, StringComparison.Ordinal), $"{rid} with {file}: {output}");
            }
        }
    }

    [Fact]
    public void PackStudio_RunsTheCudaCheck_AndShipsTheNoticeAndTheLicense()
    {
        string script = File.ReadAllText(RepositoryFile("packaging", "pack-studio.sh"));
        Assert.Contains("check-cuda.sh", script, StringComparison.Ordinal);
        Assert.Contains("THIRD-PARTY-NOTICES-Cuda.md", script, StringComparison.Ordinal);
        Assert.Contains("LICENSE-NVIDIA-CUDA.txt", script, StringComparison.Ordinal);
        Assert.Contains("linux-x64|linux-arm64|win-x64)", script, StringComparison.Ordinal);
    }

    /// <summary>
    /// A published layout (<c>SHARPLIGHT_STUDIO_PUBLISH</c> = the publish directory, <c>SHARPLIGHT_STUDIO_PUBLISH_RID</c> = its runtime) has the CUDA
    /// files its runtime needs, and the other runtimes' none (packaging/check-cuda.sh).
    /// </summary>
    [Fact]
    public void PublishedLayout_HasTheCudaFilesOfItsRuntime()
    {
        string? directory = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_PUBLISH");
        string? rid = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_PUBLISH_RID");
        Assert.SkipWhen(directory is null || rid is null || OperatingSystem.IsWindows(), "no published layout given");
        var (exit, output) = Run("bash", RepositoryFile("packaging", "check-cuda.sh"), directory!, rid!);
        Assert.True(exit == 0, output);
    }
}
