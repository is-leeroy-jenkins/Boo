# Architecture

## Purpose

This page describes the high-level architecture of the Boo application.

## Overview

Boo is organized around a Streamlit application shell and provider-specific wrapper modules.

```text
Streamlit UI
    |
    |-- app.py
    |
    |-- gpt.py
    |     |-- Chat
    |     |-- Images
    |     |-- Audio
    |     |-- Embeddings
    |     |-- Files
    |     |-- Vector Stores
    |
    |-- gemini.py
    |     |-- Chat
    |     |-- Images
    |     |-- Files
    |     |-- File Search Stores
    |
    |-- grok.py
          |-- Chat
          |-- Images
          |-- Embeddings
          |-- Files
          |-- Vector Stores