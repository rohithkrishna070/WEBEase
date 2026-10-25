"""
================================================================================
MODULE: core.__init__
================================================================================
1. WHAT THIS FILE DOES:
   Marks the `core` directory as a top-level Python package and exposes package
   versioning and sub-module namespaces for the SmartWeb Access decision engine.

2. WHY THIS FILE EXISTS (RATIONALE):
   - Conceptual Rationale: Clean architectural boundary isolating decision algorithms,
     database operations, and data models from Streamlit presentation layers.
   - Technical Rationale: Allows relative and absolute imports (e.g. `from core.models import ...`)
     across tests, CLI scripts, and multi-page Streamlit dashboards.

3. HOW IT WORKS CONCEPTUALLY:
   Acts as the central business-logic and soft-computing root for WebEase/SmartWeb Access.

4. HOW IT WORKS TECHNICALLY:
   Exposes standard `__version__` string metadata and initializes package namespace.
================================================================================
"""

__version__ = "1.0.0"

