namespace Sharplight.Studio.Models;

/// <summary>A licence the user accepted before a download: for which model, which licence, which version of its terms and when.</summary>
/// <param name="Key">What was accepted: a catalog entry id, <c>feed:</c> and the feed package id, or <c>hub:</c> and the Hub repository.</param>
/// <param name="License">The licence name.</param>
/// <param name="TermsVersion">A fingerprint of the licence and the notice shown; a changed licence or notice asks again.</param>
/// <param name="AcceptedAt">When the user accepted.</param>
internal sealed record AcceptedLicense(string Key, string License, string TermsVersion, DateTimeOffset AcceptedAt);
