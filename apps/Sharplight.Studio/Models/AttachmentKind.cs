namespace Sharplight.Studio.Models;

/// <summary>What an attachment is.</summary>
internal enum AttachmentKind
{
    /// <summary>A picture, sent to the model's vision tower.</summary>
    Image,

    /// <summary>A document whose extracted text goes into the prompt.</summary>
    Document,
}
