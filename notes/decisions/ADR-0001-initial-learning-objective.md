# ADR-0001: Initial learning objective

## Context

ChessSLM needs a minimal supervised learning task that is simple enough
to understand and debug before introducing transformers or language modeling.

Each PGN game can be replayed into pairs of:

position -> played move

## Decision

The initial model will learn:

Given a chess board position, predict the move played in the dataset.

For v0:

- Input contains only the 64 board squares.
- Target uses UCI move notation.
- Side to move, castling rights, en passant state, and move counters are
  intentionally excluded initially.

## Why

This keeps the first representation small and explicit.

The goal of v0 is not perfect chess-state representation.
The goal is to build and understand the complete supervised learning pipeline.

UCI is preferred over SAN initially because it explicitly identifies source
and destination squares.

Example:

SAN:
Nf3

UCI:
g1f3

## Consequences

Some positions cannot be represented completely.

Different legal chess states may produce the same 64-square representation.

The model will therefore have an information limitation by design.

This is acceptable for the initial baseline.

## Revisit when

Revisit after the first model can:

- train successfully,
- intentionally overfit a tiny dataset,
- and produce a reproducible baseline.