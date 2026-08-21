# Project Scope

## Project Goal

Build a service that accepts one English academic paper PDF and provides
evidence-grounded answers to Korean or English questions based on the paper's
extractable text.

## Supported Input

- English academic paper PDF
- Text-extractable PDF
- One paper per session
- One PDF upload at a time
- Maximum file size: 20MB

## Supported Questions

- Korean questions
- English questions

## Supported Answers

- Korean answers
- English answers
- The answer language follows the question language

## Evidence

Each answer should include, whenever available:

- Page number
- Section name
- Original English evidence text
- Chunk ID

## Supported Content

- Paper title
- Abstract
- Main body text
- Section headings
- Text-based table content when extractable
- Figure captions when extractable as text

## Core Features

- PDF upload
- Paper overview
- Multilingual question answering
- Evidence display

## Not Supported in the MVP

- Scanned PDFs
- OCR
- Image or graph understanding
- Complete mathematical formula understanding
- Multiple papers at the same time
- Cross-paper comparison
- Korean, Chinese, or Japanese paper input
- User login and account management
- Large-scale paper search
- Kubernetes and microservices
- Multiple LLM routing
- Payment and subscription features

## Answering Policy

- The system should answer only based on the uploaded paper.
- The system should not confidently use unsupported external knowledge.
- If the paper does not contain sufficient evidence, the system should abstain.
- The system should explain when an answer cannot be verified from the paper.