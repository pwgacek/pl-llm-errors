# Copilot Instructions for Python Code Generation

## General Principles
- Always write clean, readable, and well-structured Python code.
- Follow PEP 8 style guidelines.
- Prefer simplicity and clarity over clever or complex solutions.
- Use meaningful variable and function names.
- Add type hints wherever appropriate.

## Code Structure
- Break logic into small, reusable functions.
- Avoid deeply nested code; prefer early returns.
- Use classes only when they provide clear value.

## Type Safety and Function Signatures
- ALWAYS specify type hints for all function arguments and return values.
- NEVER omit a return type, even if it is None.
- Use explicit types (e.g., `list[str]`, `dict[str, int]`) instead of generic ones when possible.
- Avoid using `Any` unless absolutely necessary.

**Examples:**
```python
def get_user(id: int) -> dict[str, str]:
    ...

def process(items: list[int]) -> None:
    ...

## Documentation
- Do NOT document code
- Include inline comments only when necessary to explain non-obvious logic.

## Error Handling
- Use proper exception handling (try/except).
- Avoid bare except statements.
- Provide informative error messages.

## Libraries and Dependencies
- Prefer standard library solutions first.
- If external libraries are needed, choose widely used and maintained ones.
- Clearly import only what is necessary.

## Performance
- Avoid premature optimization.
- Use efficient data structures (e.g., sets for lookups, lists for ordered data).

## Security
- Never expose secrets (API keys, passwords).
- Validate and sanitize user input.

## Output Expectations
- Return complete, runnable code unless otherwise specified.
- Include example usage when helpful.

## Formatting
- Use consistent indentation (4 spaces).
- Avoid unnecessary blank lines.
- Keep line length reasonable (~88 characters).

## When Uncertain
- Make reasonable assumptions and document them.
- Prefer explicit behavior over implicit.