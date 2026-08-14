"""RAG (Retrieval-Augmented Generation) module for destination knowledge.

Provides destination knowledge retrieval and LLM-based synthesis on top of
the hotel search pipeline.  The RAG step is purely additive: hotel ranking
is unchanged, and RAG only runs when rag=true is passed to POST /search.
"""
