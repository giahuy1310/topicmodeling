---
name: read-improvise-code
description: Read and understand existing code, then propose concrete improvements, refactors, and small extensions while preserving public APIs unless explicitly asked. Use when the user asks to read, review, improve, refactor, or "improvise" on existing code.
---

# Read and Improvise Code

## When to Use This Skill

Use this skill when:

- The user shares existing code and asks to:
  - "read", "understand", "explain" or "summarize" it
  - "improve", "clean up", "refactor" or "optimize" it
  - "improvise" on it by extending functionality in a similar style
- The user wants better structure, performance, readability, or robustness **without changing public APIs** unless they say so.

Keep responses **concise by default**, and only go deeper when the user asks.

## Core Principles

1. **No hallucinations**
   - Do not invent behavior, data, or APIs that are not present in the code or clearly specified by the user.
   - If behavior is unclear or impossible to infer, say **"I don't know"** and state what information is missing.

2. **Preserve public APIs**
   - Do **not** change function/method signatures, exported symbols, or externally visible behavior unless the user explicitly asks.
   - Internal helpers, implementation details, and private functions may be refactored freely if behavior is preserved.

3. **Prefer minimal safe changes**
   - Fix correctness and robustness issues first.
   - Then improve readability, structure, and performance where it is clearly beneficial.
   - Avoid large rewrites unless the user explicitly requests them.

4. **Be language-aware**
   - Assume code is in **English** and written in common languages (Python, JS/TS, etc.).
   - Follow idioms and best practices of the detected language.
   - When adding new code, default to **Python** unless the surrounding context clearly uses another language.

5. **Concise communication**
   - Default to short, information-dense explanations.
   - Only expand with more detail, alternatives, or teaching when the user asks.

## Workflow

### 1. Understand the Context

- Identify:
  - Programming language
  - File(s) and functions/classes involved
  - Any tests or usage examples nearby
- Use the appropriate tools (e.g. `Read`, `Grep`, `SemanticSearch`) to:
  - Read the relevant file(s)
  - Find where the code is used, if needed

### 2. Summarize the Code

Produce a short summary before suggesting changes:

- What the code does
- How it is organized (key functions/classes)
- Any obvious constraints or assumptions

Keep this to a few sentences unless the user asks for a deeper explanation.

### 3. Analyze for Issues and Opportunities

Check for:

- **Correctness and edge cases**
  - Incorrect logic, off-by-one errors
  - Missing checks for `None`/null, empty inputs, or invalid data
- **Error handling**
  - Unhandled exceptions
  - Silent failures
- **Readability and structure**
  - Long functions or deeply nested logic
  - Repeated code that could be extracted
  - Poor naming or unclear responsibilities
- **Performance**
  - Obvious inefficiencies (e.g. unnecessary repeated work, poor data structures)
  - Only suggest non-trivial optimizations when clearly justified
- **Style and consistency**
  - Inconsistent patterns within the same file/module
  - Violations of dominant style in the project

### 4. Plan Improvements

Before changing code, decide:

- Which changes are **required** (bugs, correctness, severe maintainability issues)
- Which are **optional** (style, micro-optimizations)

Respect:

- **User’s constraints and preferences** (e.g. keep APIs stable, short answers)
- The existing style of the project where possible

If there are trade-offs (e.g. performance vs readability), briefly state them.

### 5. Propose Concrete Changes

When proposing changes:

- For **existing files**, use code references or small focused snippets rather than huge blocks.
- Keep diffs minimal and clearly scoped to the identified issues.
- Maintain public function/method signatures and module exports unless explicitly instructed otherwise.
- When adding new helpers or internal functions, choose clear, descriptive names.

If multiple approaches are possible:

- Recommend a default approach.
- Optionally mention 1 alternative with its trade-offs (only when useful).

### 6. Validate and Cross-Check

Whenever possible:

- Look for existing tests and see how changes would affect them.
- Suggest new tests or test cases for:
  - Edge cases
  - Previously failing or risky behavior
- Run or suggest running:
  - Linters/formatters
  - Test suites or specific test files

If you cannot be sure about a behavior change, say so explicitly.

### 7. Improvise Safely

When the user asks to "improvise" on code:

- Stay **close to the existing style and abstractions**.
- Examples of safe improvisation:
  - Adding closely related helper functions
  - Extending a class with a small, coherent method
  - Providing alternative implementations for the same API (e.g. a more efficient version)
- Always:
  - Clearly separate **original behavior** from **new improvisations** in your explanation.
  - Make sure improvisations are optional or additive unless the user asks to change behavior.

## Examples

### Example 1: Improve a Function

**User request**

> Improve this Python function for clarity and robustness without changing its signature.

**Assistant pattern**

1. Briefly summarize what the function does.
2. List specific issues (e.g. unclear variable names, missing edge-case handling).
3. Propose a revised version of the function that:
   - Keeps the same name and parameters
   - Improves readability and safety
4. Optionally mention suggested tests.

### Example 2: Improvise on Existing Code

**User request**

> Read this class and improvise: add a small feature that fits naturally.

**Assistant pattern**

1. Summarize the class responsibility.
2. Identify a natural, small extension (e.g. a convenience method or option).
3. Propose the new method or logic, ensuring:
   - Existing public behavior remains unchanged.
   - New behavior is clearly documented in the explanation.
4. Suggest how to test or validate the new feature.

---