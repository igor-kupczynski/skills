# Composition

Load this file when the task is about XRDs, XRs, Compositions, composition
functions, EnvironmentConfigs, composition revisions, or local rendering.

## Build Flow

1. Define the API with a `CompositeResourceDefinition`.
2. Install the function the Composition will call.
3. Author a `Composition` with `spec.mode: Pipeline`.
4. Create or review the XR that uses the Composition.
5. Use `crossplane render` for local previews before applying changes.

## Core Distinctions

- XRD: Define the XR schema, scope, group, names, versions, and OpenAPI schema.
- XR: Instance of the custom API defined by the XRD.
- Composition: Map an XR to one or more desired resources through a function
  pipeline.
- CompositionRevision: Snapshot a Composition so XRs can roll forward or stay
  pinned.
- EnvironmentConfig: Provide per-XR in-memory environment data through
  composition functions.

## Authoring Rules

- Keep `XRD.metadata.name` equal to `<plural>.<group>`.
- Make `Composition.spec.compositeTypeRef` match the XRD's served API version
  and kind.
- Prefer `mode: Pipeline`.
- Use `function-patch-and-transform` for straightforward field copies and
  transforms.
- Use Go or Python functions when the logic needs loops, conditionals, richer
  data shaping, or custom code.
- Use `compositionRef` for an exact Composition, `compositionSelector` for
  label-based selection, and `compositionUpdatePolicy` plus
  `compositionRevisionRef` when rollout control matters.

## Review Checklist

- Verify XRD `group`, `names`, `scope`, and `versions` are internally
  consistent.
- Verify the schema lives under `versions[].schema.openAPIV3Schema`.
- Verify every function referenced in the pipeline is installed and named
  correctly.
- Verify field paths and patch targets exist.
- Verify selectors or `matchControllerRef` are intentional for composed
  resource wiring.
- Verify revision behavior is explicit when controlled rollout matters.
- Verify EnvironmentConfig usage goes through composition functions, especially
  in modern docs.

## Minimal Example

A bare-bones XRD and Pipeline-mode Composition:

```yaml
apiVersion: apiextensions.crossplane.io/v1
kind: CompositeResourceDefinition
metadata:
  name: xdatabases.example.org
spec:
  group: example.org
  names:
    kind: XDatabase
    plural: xdatabases
  versions:
    - name: v1alpha1
      served: true
      referenceable: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                region:
                  type: string
---
apiVersion: apiextensions.crossplane.io/v1
kind: Composition
metadata:
  name: xdatabases.example.org
spec:
  compositeTypeRef:
    apiVersion: example.org/v1alpha1
    kind: XDatabase
  mode: Pipeline
  pipeline:
    - step: patch-and-transform
      functionRef:
        name: function-patch-and-transform
      input:
        apiVersion: pt.fn.crossplane.io/v1beta1
        kind: Resources
        resources:
          - name: rds-instance
            base:
              apiVersion: rds.aws.m.upbound.io/v1beta1
              kind: Instance
              spec:
                forProvider:
                  region: us-east-1
```

## Source Pages

- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/composition/composite-resource-definitions.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/composition/composite-resources.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/composition/compositions.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/composition/composition-revisions.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/composition/environment-configs.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/guides/function-patch-and-transform.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/guides/write-a-composition-function-in-go.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/guides/write-a-composition-function-in-python.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/cli/command-reference.md`
