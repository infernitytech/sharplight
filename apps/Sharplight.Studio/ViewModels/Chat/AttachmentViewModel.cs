using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Chat;

/// <summary>An attachment in the composer or in a sent message.</summary>
internal sealed class AttachmentViewModel
{
    public AttachmentViewModel(AttachmentRecord record, int estimatedTokens = 0)
    {
        Record = record;
        EstimatedTokens = record.TokenCount > 0 ? record.TokenCount : estimatedTokens;
    }

    /// <summary>The attachment.</summary>
    public AttachmentRecord Record { get; }

    /// <summary>File name.</summary>
    public string Name => Record.Name;

    /// <summary>True for images.</summary>
    public bool IsImage => Record.Kind == AttachmentKind.Image;

    /// <summary>True for documents.</summary>
    public bool IsDocument => Record.Kind == AttachmentKind.Document;

    /// <summary>Image bytes for the thumbnail.</summary>
    public byte[]? ImageData => Record.Data;

    /// <summary>Tokens the document takes (counted when sent, estimated before).</summary>
    public int EstimatedTokens { get; }

    /// <summary>Second line of the chip.</summary>
    public string Detail
    {
        get
        {
            if (IsImage)
            {
                return ByteFormat.Format(Record.Data?.LongLength ?? 0);
            }

            var tokens = $"{ByteFormat.Count(EstimatedTokens)} tokens";
            return Record.Truncated ? tokens + " · truncated" : tokens;
        }
    }
}
