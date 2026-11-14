namespace Sharplight.Models;

/// <summary>The model's weights are stored in a format (a quantization type or method) that Sharplight cannot read or compute with.</summary>
public sealed class UnsupportedWeightFormatException : NotSupportedException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="weightFormat">The format's name (e.g. <c>Q8_K</c>, <c>gptq</c>, <c>ggml type id 99</c>).</param>
    /// <param name="message">The full message.</param>
    public UnsupportedWeightFormatException(string weightFormat, string message)
        : base(message)
    {
        WeightFormat = weightFormat;
    }

    /// <summary>The format's name.</summary>
    public string WeightFormat { get; }
}
