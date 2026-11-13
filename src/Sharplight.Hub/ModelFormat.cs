namespace Sharplight.Hub;

/// <summary>The file format a Hub repository provides a model in.</summary>
public enum ModelFormat
{
    /// <summary>GGUF files (llama.cpp), usually one file per quantization, possibly split into shards.</summary>
    Gguf,

    /// <summary>A Hugging Face checkpoint: <c>config.json</c>, tokenizer files and <c>*.safetensors</c> weights.</summary>
    Safetensors,

    /// <summary>ONNX graphs (<c>*.onnx</c>, optionally with external data files).</summary>
    Onnx,
}
