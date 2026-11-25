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

    /// <summary>
    /// A Diffusers pipeline: <c>model_index.json</c> and one subfolder per component (<c>unet</c>, <c>vae</c>, <c>text_encoder</c>, ...).
    /// Listed and filtered; <see cref="ModelFileSelector"/> does not pull it (yet).
    /// </summary>
    Diffusers,

    /// <summary>
    /// whisper.cpp ggml files (<c>ggml-*.bin</c>), one file per model size and quantization. Listed and filtered; not pulled (yet).
    /// </summary>
    WhisperCpp,

    /// <summary>Sharplight packages (<c>*.slm</c>). Listed and filtered; not pulled (yet).</summary>
    Slm,

    /// <summary>
    /// PyTorch checkpoints: a Hugging Face checkpoint with <c>pytorch_model*.bin</c> weights, a checkpoint folder in its authors' layout
    /// (<c>.pth</c> / <c>.pt</c> / <c>.ckpt</c> files with their configuration) or a single <c>.ckpt</c> / <c>.pth</c> file. Read with the
    /// restricted unpickler, which never runs code. A repository that also has safetensors weights is pulled as <see cref="Safetensors"/>.
    /// </summary>
    PyTorch,
}
