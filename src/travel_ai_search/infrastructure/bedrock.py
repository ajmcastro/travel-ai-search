"""boto3 bedrock-runtime client factory.

This module is the single source of Bedrock connectivity for all Bedrock-backed
providers (embeddings, LLM, reranking).  It is imported only when a Bedrock
provider is actually requested via environment/settings — the full system runs
locally without AWS credentials.

Credentials are never committed here.  boto3 resolves them via the standard chain:
  1. Environment variables: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
     (+ optional AWS_SESSION_TOKEN for temporary credentials)
  2. ~/.aws/credentials / ~/.aws/config (named profiles)
  3. IAM instance profile / ECS task role / Lambda execution role

Required IAM permissions (minimum):
  bedrock:InvokeModel   — BedrockEmbeddingProvider and BedrockReranker
  bedrock:Converse      — BedrockLLMProvider
"""

from __future__ import annotations

from typing import Any

import boto3


def create_bedrock_client(region_name: str = "us-east-1") -> Any:
    """Return a boto3 bedrock-runtime client for the given AWS region.

    Create once at application startup and pass the client to all Bedrock
    provider constructors — boto3 clients are thread-safe after creation.
    """
    return boto3.client("bedrock-runtime", region_name=region_name)
