# Source Map

Use the Crossplane docs repository as the source of truth:

- Master source tree:
  `https://github.com/crossplane/docs/tree/master/content/master`
- Public docs site:
  `https://docs.crossplane.io/latest/`

Switch to a versioned tree only when the user requests it or the existing
material already targets that version.

## Version Choice

- Use `https://github.com/crossplane/docs/tree/master/content/master` for the
  current docs source.
- Use the matching versioned source tree for version-specific answers:
  - `https://github.com/crossplane/docs/tree/master/content/v2.2`
  - `https://github.com/crossplane/docs/tree/master/content/v2.1`
  - `https://github.com/crossplane/docs/tree/master/content/v2.0`
  - `https://github.com/crossplane/docs/tree/master/content/v2.0-preview`
  - `https://github.com/crossplane/docs/tree/master/content/v1.20`
- Check frontmatter such as `state`, `alphaVersion`, and `betaVersion` before
  describing a feature as generally available.
- If two versions differ, say which version you are using and why.

## Section Map

- `https://github.com/crossplane/docs/tree/master/content/<version>/get-started`
  - Install Crossplane.
  - Create a first managed resource, composition, or operation.
- `https://github.com/crossplane/docs/tree/master/content/<version>/composition`
  - XRDs, XRs, Compositions, CompositionRevisions, EnvironmentConfigs.
- `https://github.com/crossplane/docs/tree/master/content/<version>/managed-resources`
  - Managed resource behavior, MRDs, MRAPs, and Usages.
- `https://github.com/crossplane/docs/tree/master/content/<version>/packages`
  - Providers, Functions, Configurations, ImageConfigs.
- `https://github.com/crossplane/docs/tree/master/content/<version>/operations`
  - Operation, CronOperation, WatchOperation.
- `https://github.com/crossplane/docs/tree/master/content/<version>/guides`
  - Patch and transform, troubleshooting, function authoring, upgrades.
- `https://raw.githubusercontent.com/crossplane/docs/master/content/<version>/cli/command-reference.md`
  - `crossplane render`, `crossplane xpkg`, and flags.
- `https://github.com/crossplane/docs/tree/master/content/<version>/api/crds`
  - Exact schema and field names for Crossplane CRDs.

## Fetching Docs

Source page URLs in this skill use `raw.githubusercontent.com` so agents can
fetch them directly. The pattern is:

```
https://raw.githubusercontent.com/crossplane/docs/master/content/<version>/<path>.md
```

Directory URLs (`/tree/master/...`) are kept for navigation reference but
cannot be fetched as raw content. To fetch a specific page from a directory,
construct the raw URL for the individual `.md` file within it.

## High-Signal Source Pages

- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/_index.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/whats-crossplane/_index.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/get-started/install.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/get-started/get-started-with-managed-resources.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/get-started/get-started-with-composition.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/get-started/get-started-with-operations.md`
