namespace Sharplight.Studio.Services.Search;

/// <summary>A structural part of a document: a PDF page, a slide, a sheet, or the text under one heading.</summary>
/// <param name="Text">The section's text.</param>
/// <param name="Heading">The heading it stands under, or empty.</param>
/// <param name="Page">The page, slide or sheet number (1-based), or 0 when the format has none.</param>
/// <param name="Line">The line of the document the text starts at (1-based), or 0 when the text is not a verbatim part of the file.</param>
internal sealed record DocumentSection(string Text, string Heading = "", int Page = 0, int Line = 0);
