# Boo Documentation Theme Install Notes

## Files Included

This package includes:

- `mkdocs.yml`
- `docs/*.md`
- `docs/stylesheets/extra.css`
- `docs/javascripts/extra.js`

## Installation

Copy the package contents into the Boo repository root. The Markdown files belong in `docs/`.
The `mkdocs.yml` file belongs at the repository root.

Then run:

```powershell
mkdocs build
```

For local review:

```powershell
mkdocs serve
```

## Notes

The theme uses Material for MkDocs with a dark-first palette, deep purple primary navigation,
cyan accent color, rounded cards, high-contrast code blocks, and styled tables/admonitions.
