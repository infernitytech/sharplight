# Design tokens

Sharplight Studio (Avalonia) and web UIs built on Sharplight share one look. Its source is two JSON files in the repository:

| File | Content |
|---|---|
| `design/tokens.json` | [W3C design tokens](https://www.designtokens.org/) (`$type`, `$value`): base and semantic colours, the colours of the dark and light themes, the accent palettes, font families, the type scale, font weights, corner radii, the spacing scale and shadows |
| `design/icons.json` | the icon set: name, set (`icon`: stroke paths on a 24 x 24 grid; `caption`: title-bar buttons on a 10 x 10 box) and SVG path data |

A generator, `tools/design-tokens/Sharplight.DesignTool`, turns them into every output. Nothing generated is edited by hand.

```sh
dotnet run --project tools/design-tokens/Sharplight.DesignTool -- generate           # rewrite the outputs
dotnet run --project tools/design-tokens/Sharplight.DesignTool -- generate --check   # CI: fail when an output is out of date
```

## Outputs

| Output | Used by |
|---|---|
| `apps/Sharplight.Studio/Themes/Colors.axaml` | Studio: `AccentBrush`, `AccentSecondaryBrush`, `AccentSoftBrush`, `AccentGradientBrush`, the semantic brushes (`SuccessBrush`, `WarningBrush`, `DangerBrush`, `InfoBrush`, `SuccessTextBrush`, `WarningTextBrush`) and the theme dictionaries `Dark` / `Light` (`AppBackgroundBrush`, `SurfaceBrush`, `TextPrimaryBrush`, ...) |
| `apps/Sharplight.Studio/Themes/Tokens.axaml` | Studio: `MonoFontFamily`, the type scale (`FontSizeH1`, `FontSizeCaption`, `FontSizeMono`, ...) and the radii (`RadiusMd`, `RadiusXl`, ...), used by `Styles.axaml` as dynamic resources |
| `apps/Sharplight.Studio/Themes/Icons.axaml` | Studio: one `StreamGeometry` per icon (`IconChat`, `CaptionClose`, ...), drawn by `Path.icon` |
| `apps/Sharplight.Studio/Models/GeneratedAccentPalettes.cs` | Studio: the accent palettes of the settings (`AccentPalette.All`); `ThemeService` swaps the accent brushes at run time |
| `src/Sharplight.DesignTokens/wwwroot/sharplight-tokens.css` | web UIs: CSS custom properties |
| `src/Sharplight.DesignTokens/wwwroot/sharplight-icons.svg` | web UIs: an SVG sprite with one `<symbol>` per icon |

Names map mechanically: the token `theme.dark.surface-alt` is the Avalonia resource `SurfaceAltBrush` (in the `Dark` theme
dictionary) and the CSS property `--sl-color-surface-alt` (under `[data-theme="dark"]`); `radius.md` is `RadiusMd` and
`--sl-radius-md`; `font.size.h1` is `FontSizeH1` and `--sl-font-size-h1`; the icon `chevron-down` of set `icon` is `IconChevronDown`
and the sprite symbol `sl-icon-chevron-down`.

## Web UIs: the `Sharplight.DesignTokens` package

The package carries the CSS and the sprite as static web assets (no code). Reference it from an ASP.NET Core or Blazor project and
link the stylesheet:

```html
<link rel="stylesheet" href="_content/Sharplight.DesignTokens/sharplight-tokens.css" />
```

```css
body { background: var(--sl-color-app-background); color: var(--sl-color-text-primary); font-family: var(--sl-font-sans); }
.card { background: var(--sl-color-surface); border: 1px solid var(--sl-color-border-subtle); border-radius: var(--sl-radius-xl); }
.badge.accent { background: var(--sl-color-accent-soft); color: var(--sl-color-accent); border-radius: var(--sl-radius-sm); }
```

- Themes: dark is the default; `data-theme="light"` (or `"dark"`) on any element switches the colours below it; without the attribute
  the page follows `prefers-color-scheme`.
- Accents: `data-accent="ocean"` (`violet`, `ocean`, `emerald`, `amber`, `rose`) sets `--sl-color-accent`, `--sl-color-accent-secondary`
  and the steps `--sl-color-accent-light-1..3` / `--sl-color-accent-dark-1..3`; `--sl-color-accent-soft` and `--sl-gradient-accent`
  follow.
- Icons: `<svg class="sl-icon"><use href="_content/Sharplight.DesignTokens/sharplight-icons.svg#sl-icon-chat" /></svg>` (stroke icons
  in the current text colour, 1em; `sl-caption-icon` for the 10 px caption set).

Inter is not bundled: load it (or let the stack fall back to `system-ui`). Studio embeds it through `Avalonia.Fonts.Inter`.

## How a change flows

1. Edit `design/tokens.json` or `design/icons.json` (for example a surface colour or a new icon).
2. Run the generator. Studio's resource dictionaries, the accent palettes and the web assets change in the same commit; CI runs
   `generate --check` and fails when someone edited a token without regenerating (or edited a generated file).
3. Studio picks it up on the next build. Web UIs pick it up with the next `Sharplight.DesignTokens` package: every green `develop`
   build publishes a prerelease to the GitHub Packages feed ([releasing.md](releasing.md#prerelease-feed)), releases go to nuget.org.

Component styles are not tokens: the rules that combine tokens into a component (a card is a surface with a subtle border and the
`xl` radius; a badge is the soft accent with the `sm` radius) live in Studio's `Themes/Styles.axaml` and in each web UI's own
stylesheet, under the same class names (`card`, `badge`, `accent`, `muted`, `caption`, ...). When such a rule changes, change it in
both; when a value changes, change the token.

Values that are still literals in Studio (not yet tokens): layout paddings and margins, a few one-off corner radii in views
(`Views/*.axaml`), the user chat bubble's asymmetric radius and the translucent semantic tints of badges and callouts.
