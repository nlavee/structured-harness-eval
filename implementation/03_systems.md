# Phase 3: System Implementations

## Objectives
- Implement System Registry.
- Implement `ClaudeSystem`, `GeminiSystem`, `CodexSystem`, `StructuredHarnessSystem`.
- Implement `subprocess` execution logic with timeouts and stdin.
- Test system command generation and execution (mocked).

## Plan
1. **Registry**: Create `glass/systems/registry.py`. ✅
2. **Claude**: `glass/systems/claude.py` (command: `claude --print ...`). ✅
3. **Gemini**: `glass/systems/gemini.py` (command: `gemini -p`). ✅
4. **Codex**: `glass/systems/codex.py` (command: `codex exec ...`). ✅
5. **Structured Harness**: `glass/systems/structured_harness.py` (command: `structured-harness ...`). ✅
6. **Tests**: `tests/test_systems.py`. ✅

## Status
✅ Phase 3 Complete. Tests passed.
