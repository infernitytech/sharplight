using System.CommandLine;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>The <c>sharplight-convert</c> command line: one subcommand per package operation.</summary>
internal static class ConvertCommandLine
{
    public static RootCommand Build(SampleRuntime runtime)
    {
        var root = new RootCommand("Creates, inspects and maintains Sharplight model packages (.slm): conversion from GGUF, Hugging Face, ONNX or "
            + "another package, precompiled artifact sets per device, stripping and compaction.");
        root.Subcommands.Add(new ConvertCommand(runtime).Build());
        root.Subcommands.Add(new InfoCommand().Build());
        root.Subcommands.Add(new PrecompileCommand(runtime).Build());
        root.Subcommands.Add(new ProfileCommand(runtime).Build());
        root.Subcommands.Add(new StripCommand().Build());
        root.Subcommands.Add(new CompactCommand().Build());
        return root;
    }
}
