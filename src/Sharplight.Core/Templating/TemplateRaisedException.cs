namespace Sharplight.Templating;

/// <summary>Thrown by the template's own <c>raise_exception(...)</c> call. <see cref="Exception.Message"/> is the text given by the template.</summary>
public sealed class TemplateRaisedException : TemplateRuntimeException
{
    /// <summary>Creates the exception with the text given to <c>raise_exception</c>.</summary>
    public TemplateRaisedException(string message) : base(message) { }

    /// <summary>The text given by the template, without position information.</summary>
    public override string Message => RawMessage;
}
