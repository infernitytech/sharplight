namespace Sharplight.Distributed.Worker;

/// <summary>A segment does not match this worker's copy of the model (missing or different weights, another op set).</summary>
internal sealed class ModelMismatchException : Exception
{
    public ModelMismatchException()
    {
    }

    public ModelMismatchException(string message) : base(message)
    {
    }

    public ModelMismatchException(string message, Exception innerException) : base(message, innerException)
    {
    }
}
