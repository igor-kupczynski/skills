# Troubleshooting

Load this file when the task is to explain a failing Crossplane install,
provider, composition, managed resource, or package.

## Debug Order

1. Inspect the resource with `kubectl describe`.
2. Read `status.conditions` and the latest events.
3. Check Crossplane logs in `crossplane-system`.
4. Check provider logs in `crossplane-system`.
5. Add debug runtime config or pause reconciliation if needed.
6. Remove finalizers only as a last-resort cleanup step.

## High-Value Checks

- Use events first. Crossplane and providers log minimally by default.
- Remember events for cluster-scoped resources are often emitted to the
  `default` namespace.
- Check whether the referenced `ProviderConfig`, function, package revision, or
  Composition actually exists and is healthy.
- Use a `DeploymentRuntimeConfig` to add `--debug` to provider pods.
- Scale Crossplane or providers to zero replicas when reconciliation itself is
  making the situation worse.
- Consider the XR circuit breaker if you see reconciliation thrashing.

## Cleanup Rules

- Removing a finalizer can leave the external resource behind. Call that out
  explicitly when proposing the cleanup.
- Prefer fixing the underlying provider or credentials problem before forcing
  deletion.

## Source Pages

- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/guides/troubleshoot-crossplane.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/guides/pods.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/get-started/install.md`
