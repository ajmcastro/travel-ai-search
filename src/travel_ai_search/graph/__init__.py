"""Graph-enhanced retrieval module (Milestone 14).

Provides an in-memory directed graph of travel entities (destinations and
airports) built from the destination knowledge base.  Graph traversal answers
structural reachability questions that vector search cannot express:

  - "What destinations are similar to Mallorca?" (SIMILAR_TO edge traversal)
  - "Which destinations can I reach from Glasgow?" (FLIES_TO edge traversal)
  - "Which airports serve Barbados?" (reverse FLIES_TO lookup)

No external graph database is required.  The graph is pure Python and is
rebuilt from the knowledge JSONL file at server startup.
"""
