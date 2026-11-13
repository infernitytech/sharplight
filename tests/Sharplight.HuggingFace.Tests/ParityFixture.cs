using Sharplight.Gguf;
using Sharplight.Models;

namespace Sharplight.HuggingFace.Tests;

/// <summary>Opens the HF checkpoint and the llama.cpp-converted GGUF once for all parity tests.</summary>
public sealed class ParityFixture : IDisposable
{
    public static readonly string HfDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hf/Qwen3.5-0.8B");
    public static readonly string GgufPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf");

    public bool Available { get; }
    public ModelSource? Gguf { get; }
    /// <summary>HF source without MTP weights: directly comparable to the GGUF (which was converted without them).</summary>
    public ModelSource? Hf { get; }
    /// <summary>HF source with default options (MTP mapped to mtp.*).</summary>
    public ModelSource? HfMtp { get; }

    public ParityFixture()
    {
        Available = Directory.Exists(HfDir) && File.Exists(GgufPath);
        if (!Available)
        {
            return;
        }

        Gguf = GgufImporter.Open(GgufPath);
        Hf = HuggingFaceImporter.Open(HfDir, new HfImportOptions { IncludeMtp = false });
        HfMtp = HuggingFaceImporter.Open(HfDir);
    }

    public void Dispose()
    {
        foreach (var s in new[] { Gguf, Hf, HfMtp })
        {
            if (s != null)
            {
                foreach (var r in s.OwnedResources)
                {
                    r.Dispose();
                }
            }
        }
    }
}
