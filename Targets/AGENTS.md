# Target Notes

## Purpose

This subtree contains planning notes about provider targets and proxy
directions. It is not loaded as runtime routing configuration.

## Ownership

Architecture/planning notes only.

## Local Contracts

- Notes here must not be treated as a model registry, provider credential
  source, certification record, or executable route policy.
- Runtime target changes belong in `config/` and the router registry with
  tests and certification evidence.
- Do not place secrets or live response payloads in these notes.

## Work Guidance

Keep target descriptions factual and link them to the relevant model/provider
IDs. Label speculative or unverified targets explicitly.

## Verification

```bash
git diff --check
```

## Child DOX Index

No child directories. `Proxiers.txt` is the current planning note.
