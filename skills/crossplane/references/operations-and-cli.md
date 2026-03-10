# Operations And CLI

Load this file when the task is about `Operation`, `CronOperation`,
`WatchOperation`, feature flags, or the `crossplane` CLI.

## Operations

- `Operation`: Run a function pipeline once to completion.
- `CronOperation`: Create Operations on a cron schedule.
- `WatchOperation`: Create Operations when watched resources change.
- Enable operations with `--enable-operations` on Crossplane.
- Treat Operations as job-like workflows, not continuous reconciliation.
- Verify the function supports operation capability before using it. Check
  current docs for supported functions.

## Operation Rules

- Use `retryLimit` when transient failure handling matters.
- Use `credentials` to pass Secrets into a pipeline step.
- Use `concurrencyPolicy` on `CronOperation` and `WatchOperation` to control
  overlapping runs.
- Remember `WatchOperation` injects the changed resource as
  `ops.crossplane.io/watched-resource`.
- Remember `CronOperation` schedules use the cluster time zone.

## CLI Rules

- Use `crossplane render` to preview composition output locally.
- Use `--observed-resources`, `--extra-resources`, `--context-files`, and
  `--context-values` to simulate cluster inputs for rendering.
- Use `--include-function-results`, `--include-context`, and
  `--include-full-xr` when debug output matters.
- Use `crossplane xpkg` commands for package build and installation workflows.

## Review Checklist

- Verify Operations are enabled before debugging missing behavior.
- Verify the pipeline references installed functions.
- Verify required resources and credentials match the function's expectations.
- Verify cron schedules, concurrency policy, and history limits are deliberate.
- Verify watch filters are narrow enough to avoid noisy re-execution.

## Minimal Example

A basic Operation that runs a function pipeline once:

```yaml
apiVersion: ops.crossplane.io/v1alpha1
kind: Operation
metadata:
  name: initialize-bucket
spec:
  pipeline:
    - step: run-setup
      functionRef:
        name: function-python
      input:
        apiVersion: python.fn.crossplane.io/v1beta1
        kind: Input
        spec:
          source:
            inline: |
              def compose(req, rsp):
                  rsp.desired.resources["bucket"].resource.update({
                      "apiVersion": "s3.aws.m.upbound.io/v1beta1",
                      "kind": "Bucket",
                      "metadata": {"name": "setup-bucket"},
                      "spec": {"forProvider": {"region": "us-east-1"}}
                  })
```

## Source Pages

- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/get-started/get-started-with-operations.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/operations/operation.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/operations/cronoperation.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/operations/watchoperation.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/cli/command-reference.md`
- `https://raw.githubusercontent.com/crossplane/docs/master/content/master/get-started/install.md`
