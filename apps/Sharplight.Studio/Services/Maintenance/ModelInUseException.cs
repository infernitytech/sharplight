namespace Sharplight.Studio.Services.Maintenance;

/// <summary>A model that is resident (or being precompiled) cannot be deleted or rewritten: unload it first.</summary>
internal sealed class ModelInUseException : InvalidOperationException
{
    public ModelInUseException(string message) : base(message)
    {
    }
}
