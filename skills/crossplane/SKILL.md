---
name: Crossplane
description: Use when designing, drafting, reviewing, or troubleshooting Crossplane resources (XRDs, Compositions, managed resources, operations, providers, CLI) with the upstream docs as source of truth.
---

# Crossplane

Use this skill to work from the Crossplane docs source and published docs
without loading the entire docs tree into context.

> Last updated: 2026-03-10. Source: [docs.crossplane.io](https://docs.crossplane.io/latest/) and [crossplane/docs](https://github.com/crossplane/docs).

## Choose Sources

1. Start with [references/source-map.md](references/source-map.md).
2. Load one task reference first:
   - [references/composition.md](references/composition.md)
   - [references/resources-and-packages.md](references/resources-and-packages.md)
   - [references/operations-and-cli.md](references/operations-and-cli.md)
   - [references/troubleshooting.md](references/troubleshooting.md)
3. Read the matching Crossplane docs page or GitHub source page linked from the
   reference file only when you need exact examples, feature-state notes, or
   edge-case details.
4. Inspect the linked CRD YAML source before asserting exact field names,
   schema details, or API versions.

## Work Rules

- Default to the `master` docs source in the Crossplane docs repository.
- Switch to `v2.2`, `v2.1`, `v2.0`, `v2.0-preview`, or `v1.20` source pages
  only when the user asks for that version or the material under review already
  targets it.
- Name the Crossplane version explicitly when behavior depends on alpha, beta,
  preview, or version-specific docs.
- Prefer `mode: Pipeline` compositions and function-based workflows.
- Treat `function-patch-and-transform` as the simple option. Use Go or Python
  functions when the workflow needs loops, conditionals, richer logic, or
  reusable code.
- Prefer modern managed resource APIs such as `*.m.crossplane.io` when the docs
  support them. Call out legacy `*.crossplane.io` examples when they appear.
- Use a stable semantic version or `latest` for package image tags — don't
  copy verbatim tags from docs examples unless the user asks for them.
- When drafting manifests, include the minimum runnable object shape:
  `apiVersion`, `kind`, `metadata`, and required `spec`.
- When reviewing manifests, check Crossplane-specific linkages first:
  XRD to Composition, Composition to Function, MR to ProviderConfig,
  Operation to function capability, and MRD to MRAP.

## Deliver

- Prefer concrete manifests, `kubectl` or `helm` commands, and short
  explanations of why each object exists.
- Cite exact GitHub or `docs.crossplane.io` URLs when directing the user to the
  authoritative docs page.
