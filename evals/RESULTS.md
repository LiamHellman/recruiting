# Eval results

One row per eval run that matters: every new prompt or model version, and the baseline.
Rows are appended by the eval scripts with `--record`; add context in the note column.

## Posting parser (`evals/run_posting_eval.py`)

Golden set: `evals/postings/` (5 fictional postings with hand labels). P/R = precision / recall,
micro-averaged over postings. "Any list" ignores must vs nice; placement is the share of
correctly found skills put in the right list. Quote rejections are items dropped because
their quote was not in the posting.

| Date | Eval | Prompt | Model | Postings | Must P / R | Nice P / R | Any list P / R | Placement | Seniority | Quote rejections | Cost | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | posting | posting_parser/v1 | claude-opus-5-5 | 5 | 87% / 81% | 83% / 100% | 91% / 94% | 94% | 100% | 0% | $0.0781 | baseline |

## CV extractor (`evals/run_cv_eval.py`)

Golden set: `evals/cvs/` (10 fictional CVs with hand labels). P/R = precision / recall of
the skills the CV claims, micro-averaged over CVs; unmapped terms count by name. Dates is the
share of labelled skills whose latest `date_last_seen` is right. Quote rejections are
extracted items dropped by the quote validator. Leaks are fragments of the candidate's name
or email found in the text sent to the model; anything above 0 is a bug.

| Date | Eval | Prompt | Model | CVs | Claimed P / R | Dates | Quote rejections | Leaks | Cost | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | cv | cv_extractor/v1 | claude-opus-5-5 | 10 | 99% / 99% | 100% | 0% | 0 | $0.2071 | baseline |

## README extractor (`evals/run_readme_eval.py`)

Golden set: `evals/readmes/` (8 fictional READMEs with hand labels, including a prompt
injection, a roadmap of technologies not used, and author names and emails to redact).
P/R = precision / recall of the technologies the project uses, micro-averaged; unmapped terms
count by name. Quote rejections and leaks as for the CV extractor.

| Date | Eval | Prompt | Model | READMEs | P / R | Quote rejections | Leaks | Cost | Note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-03 | readme | readme_extractor/v1 | claude-opus-5-5 | 8 | 98% / 98% | 0% | 0 | $0.0340 | baseline; a run minutes earlier gave 95% / 98% ($0.0400): misses vary between runs (AWS written as "AWS ECS"; "npm", "integration tests" extra) |
