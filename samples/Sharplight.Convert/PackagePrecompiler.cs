using System.Diagnostics;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>
/// Compiles every graph of a package on one device and appends the artifact sets the compilation recorded. Language models are
/// compiled the way <see cref="LlmModel"/> compiles them at load (so the stored kernels are the ones a later load asks for);
/// other graphs are compiled with the given symbol bounds.
/// </summary>
internal sealed class PackagePrecompiler(TextWriter output)
{
    private const string TemporaryCachePrefix = "sharplight-precompile-";

    public void Run(PrecompileSettings settings)
    {
        // A fresh kernel cache directory keeps the driver pipeline cache free of other models' pipelines.
        var temporaryCache = settings.UseUserKernelCache ? null : Directory.CreateTempSubdirectory(TemporaryCachePrefix).FullName;
        try
        {
            using var device = OpenDevice(settings.Device, temporaryCache);
            if (device.ArtifactTargets.Count == 0)
            {
                throw new SampleException($"{device.Info} produces no precompiled artifacts.", ExitCodes.InvalidInput);
            }

            output.WriteLine($"Compiling {settings.PackagePath} on {device.Info}");
            if (IsLanguageModel(settings.PackagePath))
            {
                CompileLanguageModel(settings, device);
            }
            else
            {
                CompileGraphs(settings, device);
            }
        }
        finally
        {
            if (temporaryCache is not null)
            {
                Directory.Delete(temporaryCache, recursive: true);
            }
        }
    }

    private static IComputeDevice OpenDevice(DeviceSpec spec, string? kernelCacheDirectory)
    {
        var providers = ProviderRegistry.Discover();
        var provider = providers.FirstOrDefault(p => p.Provider.Name == spec.Provider);
        if (provider.Provider is null)
        {
            var known = string.Join(", ", providers.Select(p => p.Provider.Name));
            throw new SampleException($"Unknown provider '{spec.Provider}'; available: {known}.", ExitCodes.InvalidInput);
        }

        if (!provider.Availability.IsAvailable)
        {
            throw new SampleException($"Provider '{spec.Provider}' is not available: {provider.Availability.Reason}");
        }

        var info = SampleRuntime.GetDevice(provider.Provider, spec.Index);
        return provider.Provider.OpenDevice(info, new DeviceOptions { KernelCacheDirectory = kernelCacheDirectory });
    }

    private static bool IsLanguageModel(string path)
    {
        var source = new ModelPackageImporter().Open(path);
        try
        {
            return source.Definition!.Graphs.ContainsKey(WellKnown.DecodeGraph);
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    private void CompileLanguageModel(PrecompileSettings settings, IComputeDevice device)
    {
        using var model = LlmModel.Load(settings.PackagePath, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            ContextLength = settings.ContextLength,
            MaxBatchTokens = settings.BatchTokens,
            Log = output.WriteLine,
        });
        foreach (var graph in new[] { WellKnown.DecodeGraph, WellKnown.DecodeMultimodalGraph }.Where(model.Definition.Graphs.ContainsKey))
        {
            using var session = model.CreateSession(graph);
        }

        if (model.Definition.Graphs.ContainsKey(WellKnown.VisionEncodeGraph))
        {
            using var encoder = model.CreateVisionEncoder();
        }

        // Entries served from the package's own sets point into its mapping: store them before the model is disposed.
        Store(settings.PackagePath, model.ExportCompilation(settings.Tiers));
    }

    private void CompileGraphs(PrecompileSettings settings, IComputeDevice device)
    {
        var source = new ModelPackageImporter().Open(settings.PackagePath);
        try
        {
            var artifacts = new CompilationArtifacts(source.CompilationCaches);
            var compileOptions = new CompileOptions
            {
                MaxTokens = settings.BatchTokens,
                ContextLength = settings.ContextLength,
                SymbolBounds = settings.SymbolBounds,
                CacheFor = artifacts.For,
            };
            foreach (var (name, graph) in source.Definition!.Graphs)
            {
                var stopwatch = Stopwatch.StartNew();
                using var compiled = Compile(name, graph, device, compileOptions);
                output.WriteLine($"Compiled '{name}' in {stopwatch.ElapsedMilliseconds} ms");
            }

            Store(settings.PackagePath, artifacts.Export(settings.Tiers));
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    private static CompiledModel Compile(string name, Graph.ComputeGraph graph, IComputeDevice device, CompileOptions options)
    {
        try
        {
            return CompiledModel.Compile(graph, new PlacementOptions { Devices = [new DeviceSlot { Device = device }], Compile = options });
        }
        catch (InvalidOperationException error) when (error.Message.StartsWith("No bound", StringComparison.Ordinal))
        {
            throw new SampleException($"Graph '{name}': {error.Message} Give it with --bound SYMBOL=VALUE.", ExitCodes.InvalidInput);
        }
    }

    private void Store(string path, IReadOnlyList<CompilationCache> recorded)
    {
        // A set served entirely from sets already in the package adds nothing (code entries are content addressed).
        var sets = recorded.Where(s => s.FallbackHits < s.Entries.Count).ToList();
        foreach (var unchanged in recorded.Except(sets))
        {
            output.WriteLine($"Already stored: {ArtifactDescriptor.Format(unchanged.Descriptor)} ({unchanged.Entries.Count} entries)");
        }

        if (sets.Count == 0)
        {
            output.WriteLine("Nothing new to store.");
            return;
        }

        var stopwatch = Stopwatch.StartNew();
        ModelPackage.AddArtifacts(path, sets);
        foreach (var set in sets)
        {
            long bytes = set.Entries.Values.Sum(e => (long)e.Length);
            output.WriteLine($"Stored {ArtifactDescriptor.Format(set.Descriptor)}: {set.Entries.Count} entries, {ByteSize.Format(bytes)} "
                + $"({set.FallbackHits} already in the package)");
        }

        output.WriteLine($"{path}: {ByteSize.Format(new FileInfo(path).Length)} [{stopwatch.ElapsedMilliseconds} ms]");
    }
}
