# Resources And Packages

Load this file when the task is about managed resources, providers,
ProviderConfigs, MRDs, MRAPs, Usages, Functions, Configurations, or
ImageConfigs.

## Managed Resources

- Treat `spec.forProvider` as the source of truth for the external resource.
- Use `initProvider` for write-once values that should apply only at creation.
- Use direct external-name fields, `nameRef`, `selector`, or
  `matchControllerRef` depending on how tightly resources should bind.
- Expect immutable provider fields to require delete and recreate outside normal
  reconciliation. Crossplane does not recreate automatically just because a
  field changed.
- Distinguish namespaced modern APIs such as `*.m.crossplane.io` from legacy
  cluster-scoped `*.crossplane.io` APIs.

## Managed Resource Activation

- In modern v2 docs, providers can be converted into `ManagedResourceDefinition`
  objects during installation.
- Use `ManagedResourceActivationPolicy` to activate only the resources needed.
- Treat MRAP wildcards as prefix-only matching. `*.s3.aws.m.crossplane.io`
  works; arbitrary regex-style patterns do not.
- Remember MRD activation is one-way from `Inactive` to `Active`.

## Packages

- Install providers with `pkg.crossplane.io/v1` `Provider`.
- Install composition functions with `pkg.crossplane.io/v1` `Function`.
- Install reusable platform bundles with `pkg.crossplane.io/v1`
  `Configuration`.
- Use `packagePullPolicy`, `revisionActivationPolicy`, and
  `revisionHistoryLimit` intentionally for package lifecycle control.
- Use `ImageConfig` for pull secrets, signature verification, or runtime config
  overrides based on image prefix matches.
- When multiple `ImageConfig` objects match an image, the longest prefix wins.

## Usages

- Use `Usage` to prevent deletion of an in-use resource or enforce deletion
  ordering.
- Use selectors carefully because they resolve once and then persist the chosen
  `resourceRef.name`.
- Combine labels with `matchControllerRef` when the resource must stay within
  one composition instance.

## Review Checklist

- Verify the provider or function package is installed before reviewing the
  dependent resource.
- Verify the right `ProviderConfig` or `ClusterProviderConfig` is referenced.
- Verify `forProvider` fields belong to the specific provider schema being used.
- Verify modern versus legacy API groups are not mixed accidentally.
- Verify MRAP patterns activate the intended MRDs and nothing broader.
- Verify `ImageConfig` prefix scope and precedence.

## Minimal Example

A Provider install, ProviderConfig, and managed resource:

```yaml
apiVersion: pkg.crossplane.io/v1
kind: Provider
metadata:
  name: provider-aws-s3
spec:
  package: xpkg.upbound.io/upbound/provider-aws-s3:v1.20.0
---
apiVersion: aws.upbound.io/v1beta1
kind: ProviderConfig
metadata:
  name: default
spec:
  credentials:
    source: Secret
    secretRef:
      namespace: crossplane-system
      name: aws-creds
      key: credentials
---
apiVersion: s3.aws.m.upbound.io/v1beta1
kind: Bucket
metadata:
  name: my-bucket
  namespace: default
spec:
  forProvider:
    region: us-east-1
  providerConfigRef:
    name: default
```

## Source Pages

- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/get-started/get-started-with-managed-resources.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/managed-resources/managed-resources.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/managed-resources/managed-resource-definitions.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/managed-resources/managed-resource-activation-policies.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/managed-resources/usages.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/packages/providers.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/packages/functions.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/packages/configurations.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/packages/image-configs.md`
