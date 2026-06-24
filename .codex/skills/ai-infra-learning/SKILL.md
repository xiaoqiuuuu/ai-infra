---
name: ai-infra-learning
description: Use only for this repository's AI Infra learning workflow when the user asks to start learning, continue studying, plan or review a stage, create stage tasks, check knowledge_map.md progress, generate acceptance questions, explain roadmap topics, or advance through ai_infra_learning_roadmap.md for AI Infra, LLM serving, training infrastructure, GPU systems, distributed training, vLLM, SGLang, TensorRT-LLM, CUDA, Triton, NCCL, or cluster engineering.
---

# AI Infra Learning

## Scope

Use this skill only inside the AI Infra learning repository whose root contains `ai_infra_learning_roadmap.md`.

Treat `ai_infra_learning_roadmap.md` as the source of truth for the learning path. The current repository state is the progress record. If there are no stage directories, treat progress as 0 and start from stage 0.

Do not turn this into a generic study coach. Keep all plans, explanations, tasks, and acceptance checks tied to the AI Infra roadmap and the target role described there.

## Repository Workflow

When the user asks to start or continue learning:

1. Find the repository root by locating `ai_infra_learning_roadmap.md`.
2. Read the roadmap enough to identify the current or next stage.
3. Inspect existing `stage_*` directories and each stage's `knowledge_map.md`.
4. Decide whether to continue the current stage or create the next stage.
5. Give the user a concrete learning task for the current session and the exact file they should update.

Preferred stage layout:

```text
stage_00_understand_ai_infra_map/
  learning_tasks.md
  knowledge_map.md
```

Use `stage_XX_short_slug` for new directories. Keep stage numbers aligned with the roadmap.

## Creating A Stage

Create a new stage directory when:

- No stage directory exists yet.
- The current stage's `knowledge_map.md` satisfies the roadmap output requirements and acceptance questions.
- The user explicitly asks to prepare a specific stage.

For each new stage, create `learning_tasks.md` with:

- `# Stage XX: <roadmap stage title>`
- `## Goal`
- `## Tasks`
- `## Reference Materials`
- `## Output`
- `## Acceptance Questions`
- `## Next Checkpoint`

In `## Reference Materials`, include a small, curated reading list instead of a link dump:

- Prefer official documentation, original papers, project documentation, and source repositories.
- Include 4-8 resources per stage.
- Split resources into `Required` and `Optional`.
- For each resource, state what to read and what question it should help answer.
- Include direct URLs.
- Avoid paid courses, broad search queries, and generic motivational content unless the user explicitly asks for them.

Also create `knowledge_map.md` if missing, with a concise template:

```markdown
# Knowledge Map

## Core Concepts

## My Explanations

## Open Questions

## Evidence Of Completion
```

Do not overwrite a user's existing notes. If a file exists, read it and update only when the user asks for edits or when adding a clearly missing task/checkpoint.

## Reviewing Progress

When checking progress:

1. Read the current stage's `learning_tasks.md` and `knowledge_map.md`.
2. Compare the notes against the roadmap's output and acceptance requirements.
3. State whether the stage is `in progress` or `complete`.
4. If incomplete, list the smallest missing concepts or explanations.
5. If complete, create the next stage task files and tell the user where to continue.

Do not advance stages just because files exist. Advance only when the user's notes show they can explain the stage acceptance questions in their own words.

## Teaching Style

Use Chinese by default. Be direct and concrete.

Favor system explanations over motivational advice. Tie concepts back to AI Infra engineering:

- Training Infra
- Inference Infra
- GPU Compute
- Communication
- Storage
- Scheduling
- Stability

For concept explanations, use the user's roadmap as context and prefer examples from LLM serving, vLLM, PyTorch distributed training, KV Cache, DDP/FSDP, NCCL, CUDA, Triton, and GPU clusters.

For a study session, respond with:

- Today's task
- Why this matters for AI Infra
- Reference materials for this task
- What to write in `knowledge_map.md`
- Acceptance questions

Keep the session small enough to finish in one sitting.

## File Handling

Leave `ai_infra_learning_roadmap.md` unchanged unless the user explicitly asks to revise the roadmap.

Use local files as the progress source. Do not assume progress from conversation alone if stage files exist.

Prefer creating or updating stage files over long chat-only plans when the user asks to start, continue, or prepare a stage.
