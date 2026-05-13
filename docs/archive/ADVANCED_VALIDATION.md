# Advanced validation

This prototype is meant to be publishable, so it includes explicit validation hooks.

## Validation stack
1. unit tests for metrics and kernel round-tripping
2. runtime tests for selection and fallback
3. store tests for index stability
4. digest tests for tamper detection

## Why the validation matters
A publishable codebase should not merely exist. It should be able to show that:
- kernels survive save/load
- the runtime chooses the right path
- unrelated prompts fall back
- metadata is preserved consistently
