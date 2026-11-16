namespace Sharplight.Package;

/// <summary>One attachment of a package (an ATCH section, see <see cref="Models.ModelDefinition.Attachments"/>).</summary>
/// <param name="Name">Attachment name, as components refer to it (e.g. <c>tokenizer/vocab.json</c>).</param>
/// <param name="Offset">Absolute file offset of the bytes (64 KiB aligned).</param>
/// <param name="Length">Size in bytes.</param>
public sealed record PackageAttachmentInfo(string Name, long Offset, long Length);
