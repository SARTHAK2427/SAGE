You are going to perform a deep audit and controlled polishing of the current SAGE repository.

THIS IS A MULTI-STAGE TASK.

CRITICAL EXECUTION RULE:
YOU MAY PERFORM ONLY ONE STAGE AT A TIME.

After completing the currently authorized stage:
1. report what you found/did,
2. list files created/modified,
3. STOP COMPLETELY,
4. wait for me to explicitly say "proceed".

Do NOT automatically begin the next stage.
Do NOT interpret completion of one stage as authorization for another.

============================================================
ABSOLUTE ARCHITECTURE FREEZE
============================================================

The existing SAGE architecture is FROZEN.

You are NOT authorized to redesign, replace, simplify, restructure, or semantically alter:

- overall SAGE architecture
- Gemma 4B role as sole semantic/strategic controller
- agent loop/protocol
- tool inventory
- tool boundaries
- tool contracts
- deterministic socket contracts
- socket → plug mapper architecture
- document database architecture
- canonical artifact architecture
- RAG/exact-search/artifact-fetch semantics
- vision/OCR responsibility
- coder responsibility
- math responsibility
- model-routing semantics
- remote/local inference architecture
- JSON protocols
- model responsibilities
- derived-artifact lifecycle
- canonical-vs-derived data rules

Do not introduce another semantic router, controller, agent, autonomous subsystem, or hidden model decision.

Gemma remains the sole component allowed to make semantic cross-capability decisions.

Infrastructure/deterministic code may execute only decisions already authorized by the architecture.

If you discover what appears to be an architectural flaw:
DO NOT CHANGE IT.

Document it separately as:
"ARCHITECTURE REVIEW REQUIRED"

and explain:
- the issue
- evidence
- impact
- possible options

Then leave the architecture untouched.

============================================================
GENERAL CODE MODIFICATION POLICY
============================================================

During stages where code modification is explicitly authorized, you MAY improve implementation quality inside the frozen architecture.

Allowed examples:

- fix genuine bugs
- fix security vulnerabilities
- improve validation
- improve deterministic error handling
- eliminate demonstrably dead code
- remove unused imports
- eliminate genuine implementation duplication
- improve naming/readability
- simplify unnecessarily complicated implementation
- improve type safety
- improve resource cleanup
- improve timeout handling
- improve logging/observability
- improve comments/docstrings where useful
- improve test coverage
- make safe performance improvements
- fix inconsistent implementation of an already-frozen contract

NOT allowed without explicit approval:

- changing public contracts
- changing schemas
- changing model roles
- changing prompts during a code-cleanup stage
- changing tool behavior/meaning
- changing orchestration semantics
- changing mapper semantics
- merging components merely because they look redundant
- deleting compatibility code without proving it is unused
- speculative refactoring
- large-scale file/folder restructuring
- adding unnecessary abstractions
- replacing working subsystems with preferred libraries/frameworks

"Cleaner" is NOT sufficient justification for changing architecture.

Before deleting anything, prove from repository references/tests/runtime paths that it is actually dead.

Prefer small, reviewable changes over broad rewrites.

============================================================
SOURCE OF TRUTH
============================================================

Do not rely on old assumptions about SAGE.

Inspect the CURRENT repository.

The repository has changed substantially.

Determine actual behavior from:
- current source code
- current configuration
- current prompts
- current schemas
- tests
- runtime paths
- existing documentation where still accurate

When documentation disagrees with executable code, explicitly identify the discrepancy.

============================================================
OUTPUT LOCATION
============================================================

All requested audit/documentation artifacts must be written under:

C:\Users\ADMIN\Desktop\SAGE\notes\

Do not overwrite unrelated existing notes.

============================================================
TESTING RULE
============================================================

Whenever a future authorized stage modifies executable code:

1. run focused tests for the changed subsystem,
2. run the complete regression suite,
3. report exact results.

Never hide failing tests.
Never weaken/delete a test merely to make the suite pass.

Do not commit.
Do not push.

============================================================
STAGE 1 — DEEP CODEBASE AUDIT ONLY
============================================================

THIS IS THE ONLY STAGE CURRENTLY AUTHORIZED.

DO NOT MODIFY PRODUCTION CODE IN STAGE 1.

Perform a deep static/code-path analysis of the CURRENT SAGE repository.

Trace important runtime paths rather than merely scanning filenames.

Inspect at minimum:

- app/backend entrypoints
- orchestration loop
- Gemma invocation
- tool dispatcher
- all registered tools
- model manager
- model client
- local/remote model transport
- coder execution + Docker sandbox
- vision execution
- document ingestion
- document DB
- Chroma/indexing
- artifact storage/resolution
- RAG
- exact search
- deterministic schemas
- socket→plug mappers
- configuration/environment handling
- prompt loading
- error handling
- temporary files
- concurrency/state
- subprocess handling
- filesystem trust boundaries
- uploaded-file handling
- remote HTTP trust boundaries
- authentication/secrets handling
- tests
- legacy compatibility paths
- UI/backend interface where relevant

Look specifically for:

SECURITY:
- command injection
- path traversal
- unsafe file handling
- archive/path extraction issues
- Docker/sandbox escape-enabling configuration
- unsafe subprocess construction
- secret leakage
- insecure defaults
- untrusted regex/resource-exhaustion risks
- unbounded input/resource consumption
- unsafe deserialization/parsing
- remote transport trust issues
- malformed model-output handling

CORRECTNESS:
- swallowed exceptions
- fake success states
- ghost IDs
- incorrect fallback behavior
- stale state
- race conditions
- resource leaks
- incorrect model state
- malformed JSON handling
- schema mismatches
- partial/error states represented incorrectly
- provenance loss
- canonical/derived data contamination

CODE QUALITY:
- dead code
- unused imports
- duplicate implementations
- redundant compatibility layers
- unreachable branches
- confusing naming
- oversized functions
- hidden coupling
- unnecessary global state
- inconsistent error conventions
- stale comments/docs
- TODO/FIXME/HACK markers
- weak typing where it creates real risk

PERFORMANCE:
Identify obvious implementation inefficiencies, but DO NOT optimize them in this stage.

For every finding classify:

Severity:
CRITICAL / HIGH / MEDIUM / LOW / INFO

Confidence:
CONFIRMED / HIGH-CONFIDENCE / SUSPECTED

Category:
SECURITY / CORRECTNESS / RELIABILITY / PERFORMANCE /
MAINTAINABILITY / DEAD-CODE / DUPLICATION / DOCUMENTATION

For each meaningful finding provide:

- ID
- severity
- confidence
- affected file(s)
- affected symbol/function where possible
- description
- evidence
- runtime impact
- whether currently covered by tests
- recommended remediation
- whether remediation is implementation-only or requires architecture review

Create:

notes/vulnerability_analysis.md

IMPORTANT:
Despite the historical name "vulnerability_analysis", this document should cover BOTH:
security vulnerabilities AND serious implementation/code-quality problems.

Also include:

## Executive Summary

## Repository Health

## Security Findings

## Correctness Findings

## Reliability Findings

## Code Quality / Maintainability

## Dead Code Candidates

## Duplication / Redundancy Candidates

## Performance Findings Deferred to Optimization Stage

## Architecture Review Required
(if any — documentation only)

## Recommended Fix Order

## Files That Should NOT Be Touched

## Proposed Stage 1B Fix Set

The final section must propose a SMALL, SAFE first batch of implementation-only fixes.

Do NOT apply those fixes yet.

============================================================
FUTURE STAGES — NOT AUTHORIZED YET
============================================================

These exist only so you understand the eventual workflow.

DO NOT EXECUTE THEM.

STAGE 1B:
Apply the first approved vulnerability/code-quality fix batch.

STAGE 1C+:
Continue additional small approved fix batches one at a time.

STAGE 2:
Robustness + performance investigation.

This will later investigate, among other things:

- why a normal message currently spends roughly 7–14 seconds before model inference begins even when no document is uploaded
- model-switch gaps of roughly 2–4 seconds
- unnecessary blocking work
- model startup/shutdown overhead
- remote transport overhead
- redundant parsing/serialization
- unnecessary repeated work
- caching opportunities
- graceful error prevention/recovery
- observability/timing instrumentation
- pipeline bottlenecks

It will create:
notes/robustness_optimization.md

IMPORTANT:
Stage 2 begins with MEASUREMENT/ROOT-CAUSE ANALYSIS.
Do not blindly optimize based on assumptions.

STAGE 3:
Model/prompt/context audit.

This will inspect the CURRENT prompt files and actual prompt construction code.

Questions include:

- what system prompt does each model actually receive?
- which prompt files are stale relative to current SAGE?
- are system prompts resent on every inference?
- what exactly happens across Gemma tool loops?
- does each loop reconstruct the entire conversation/system prompt?
- what causes context growth?
- can invariant instructions be handled more efficiently without changing semantics?
- is Gemma appropriately synthesizing tool outputs?
- is it hallucinating/recreating information instead of grounding itself in tool results?
- would deterministic patching be appropriate anywhere, WITHOUT violating Gemma's role as sole semantic controller?
- how can prompts become shorter and more precise?

Prompt changes are NOT authorized before this stage.

STAGE 4:
High-level architecture documentation.

Create:
notes/system_architecture.md

It must document the ACTUAL current SAGE implementation and contain rich Mermaid diagrams/flowcharts covering:

- major departments/components
- request lifecycle
- document lifecycle
- agent/tool loop
- model execution
- local/remote inference
- artifact lifecycle
- canonical vs derived information
- major JSON contracts
- failure paths
- major integration boundaries

It must be useful to:
- humans
- GitHub reviewers
- AI code-analysis systems
- another developer merging SAGE with another branch

STAGE 5:
Low-level architecture/interface documentation.

Create:
notes/system_architecture_low.md

This must document the ACTUAL implementation in depth.

Use the existing SAGE "socket → plug" analogy:

SOCKET:
Everything a component is capable of exposing/outputting,
including useful data, metadata, provenance, status, timing,
identifiers, errors, etc.

PLUG:
The explicitly selected subset/shape consumed by the receiving
component.

Document component-by-component:

- responsibility
- implementation location
- inputs
- rich output socket
- receiving plug(s)
- mapper/adapter
- validation
- error states
- persistence
- relevant schemas
- actual JSON examples
- cross-component dependencies

Explicitly reference the current JSON/schema files rather than inventing contracts.

Include detailed Mermaid diagrams.

This documentation will be used to help merge the polished current
SAGE codebase with another branch containing a newer memory subsystem.

DO NOT implement memory.
DO NOT redesign SAGE around memory.
Only document integration boundaries sufficiently for another developer
to perform that merge later.

============================================================
CURRENT COMMAND
============================================================

Execute STAGE 1 ONLY.

Perform the deep audit.
Create:

notes/vulnerability_analysis.md

Do not modify production code.
Do not begin fixes.
Do not create Stage 2/3/4/5 documents.

When Stage 1 is complete:
give me a concise summary,
tell me the path of the created report,
and STOP.

Wait for my explicit instruction to proceed.