using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>Resolves <c>data_location = EXTERNAL</c> tensor data (usually a <c>.onnx.data</c> file next to the model).</summary>
internal interface IOnnxExternalDataResolver
{
    /// <summary>Opens a byte range of an external data file as a weight source.</summary>
    /// <param name="location">Relative path of the data file as stored in the tensor.</param>
    /// <param name="offset">Byte offset of the tensor data within the file.</param>
    /// <param name="length">Byte length, or -1 for "to the end of the file".</param>
    /// <exception cref="InvalidDataException">The location escapes the model directory or the range lies outside the file.</exception>
    public IWeightSource Open(string location, long offset, long length);
}
