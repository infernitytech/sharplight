namespace Sharplight.Studio.Models;

/// <summary>What an attachment is.</summary>
internal enum AttachmentKind
{
    /// <summary>A picture, sent to the model's vision tower.</summary>
    Image,

    /// <summary>A document whose extracted text goes into the prompt.</summary>
    Document,

    /// <summary>A picture the app generated for the conversation (an answer to the image command); it is shown, never sent to the language model.</summary>
    GeneratedImage,
}
