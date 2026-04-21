# pl-llm-errors
This repository contains the source code and datasets for experiments conducted as part of the Master's thesis titled: "Analysis of the impact of errors in Polish-language queries on the quality of responses from large language models". The objective of the project is to investigate the robustness of various models against Polish-specific errors.

# how to run

## Build prompts

```bash
python -m src.prompt_preparation.prompt_builder
```

## Run inference (collect LLM responses)

```bash
python -m src.inference.inference_runner
```

## Run judgement (LLM-as-a-judge)

```bash
python -m src.judgement.judgement_runner <answers_file.json>
```

---

### bielik-4.5b-v3.0-instruct

| Error type   | Correct | Incorrect | Error | Accuracy |
|--------------|--------:|----------:|------:|---------:|
| identity     |     304 |       196 |     0 |   60.80% |
| diacritic    |     301 |       199 |     0 |   60.20% |
| punctuation  |     281 |       219 |     0 |   56.20% |
| spelling     |     259 |       241 |     0 |   51.80% |
| typo 30%     |     277 |       223 |     0 |   55.40% |
| typo 70%     |     248 |       252 |     0 |   49.60% |
| typo 100%    |     240 |       260 |     0 |   48.00% |

### gemma3-4b

| Error type   | Correct | Incorrect | Error | Accuracy |
|--------------|--------:|----------:|------:|---------:|
| identity     |     279 |       221 |     0 |   55.80% |
| diacritic    |     275 |       225 |     0 |   55.00% |
| punctuation  |     260 |       240 |     0 |   52.00% |
| spelling     |     231 |       269 |     0 |   46.20% |
| typo 30%     |     241 |       259 |     0 |   48.20% |
| typo 70%     |     224 |       276 |     0 |   44.80% |
| typo 100%    |     203 |       296 |     1 |   40.60% |