# Story Analyzer AI — Architecture Specification

This repository contains the architecture, backend services, extraction engines, and frontend for **Story Analyzer AI**, an AI-powered screenplay analysis platform with strict citation grounding.

See full Phase 1 specification in the system artifacts:
- Dual Citation Requirement: Every analytical claim requires Screenplay Citation (scene/page/dialogue) + Theory Citation (book/author/concept).
- Grounding Rule: If evidence is insufficient, return: `"Insufficient information to make a reliable assessment."`

