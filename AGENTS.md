# AGENTS

This repository stores reusable skills for coding agents.
We learn together: if an agent finds something especially interesting, unexpected, or a hard-won lesson, add it here so future runs benefit.

## Conventions

- Keep each skill in its own directory under `skills/`.
- Include a `SKILL.md` file with clear instructions and YAML frontmatter (`name`, `description`) delimited by `---`.
- Keep examples small and practical.
- **When creating or updating a skill**, update the skills table in `README.md` (between `<!-- BEGIN SKILLS -->` and `<!-- END SKILLS -->` markers). Read each `skills/*/SKILL.md` frontmatter to rebuild the table.

## Lessons

- **Mistral model IDs ≠ docs URL slugs.** The Mistral docs URL for a model (e.g. `docs.mistral.ai/models/mistral-large-3-25-12`) uses a different format than the API model ID (`mistral-large-2512`). API IDs follow `{family}-{YYMM}`. Always verify model IDs against the [changelog](https://docs.mistral.ai/getting-started/changelog/) or the models list API, not the URL path.
- **Mistral gateway timeout ~90-120s.** Non-streaming chat completion requests that run longer than ~90-120 seconds get silently terminated (connection reset, not an HTTP error). Always use `stream: true` for long-running completions (large context, vision, high `max_tokens`).
