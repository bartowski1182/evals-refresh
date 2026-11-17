# Eval coverage matrix

Benchmarks reported by the model cards of 11 open-weight model families, mapped to Inspect packages, capability buckets and
execution tiers. **Repo** marks the evals implemented and run in this repository (for at least one model); their protocol
and results are in `<org>_<model>/<category>/<task>/README.md`. Sorted by how many families report the benchmark.

## Benchmarks

| # | Benchmark | Evaluates | Subclass | Exec | Repo | Where | Detail | Families reporting |
|---|---|---|---|---|---|---|---|---|
| 9 | Terminal-Bench 2.1 | agentic-and-coding | swe-agent | agent | ✅ | inspect_harbor | terminal_bench_2_1 | DeepSeek-V4, GLM-5.x, Granite-4.2, Kimi-K3, MiniCPM5, Nemotron-3.x, Nex-N2.5, Qwen3.8, Step-3.5-Flash |
| 8 | GPQA Diamond | knowledge-and-reasoning | science-reasoning | none | ✅ | inspect_evals | gpqa_diamond | DeepSeek-V4, GLM-5.x, Gemma-4, Granite-4.2, Kimi-K3, MiniCPM5, Nemotron-3.x, Qwen3.8 |
| 7 | GDPval-AA | general-tool-usage | pro-work | agent |  | inspect_evals | gdpval | DeepSeek-V4, GLM-5.x, Granite-4.2, Kimi-K3, MiniCPM5, Nemotron-3.x, Nex-N2.5 |
| 7 | HLE | knowledge-and-reasoning | expert-knowledge | none |  | inspect_evals | hle | DeepSeek-V4, GLM-5.x, Gemma-4, Kimi-K3, MiniCPM5, Nemotron-3.x, Qwen3.8 |
| 7 | LiveCodeBench | agentic-and-coding | code-gen | sandbox | ✅ | inspect_harbor | livecodebench  (+ native livecodebench_pro = different bench) | DeepSeek-V4, Gemma-4, Granite-4.2, MiniCPM5, Nemotron-3.x, Qwen3.8, Step-3.5-Flash |
| 6 | BrowseComp | general-tool-usage | search-research | agent |  | inspect_evals | browse_comp | DeepSeek-V4, Kimi-K3, MiniCPM5, Nemotron-3.x, Nex-N2.5, Step-3.5-Flash |
| 6 | SWE-bench Verified | agentic-and-coding | swe-agent | agent | ✅ | inspect_evals | swe_bench | DeepSeek-V4, Granite-4.2, MiniCPM5, Mistral-M3.5, Nemotron-3.x, Step-3.5-Flash |
| 6 | SWE-bench Pro | agentic-and-coding | swe-agent | agent | ✅ | inspect_harbor | scale_ai_swe_bench_pro / cais_swebenchpro | DeepSeek-V4, GLM-5.x, Granite-4.2, MiniCPM5, Nex-N2.5, Qwen3.8 |
| 6 | HMMT | knowledge-and-reasoning | math-reasoning | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4, GLM-5.x, Granite-4.2, MiniCPM5, Nemotron-3.x, Step-3.5-Flash |
| 5 | MMLU-Pro | knowledge-and-reasoning | knowledge-recall | none | ✅ | inspect_evals | mmlu_pro | DeepSeek-V4, Gemma-4, Granite-4.2, MiniCPM5, Nemotron-3.x |
| 5 | DeepSWE 1.1 | agentic-and-coding | swe-agent | agent | ✅ | inspect_harbor | datacurve_deep_swe_1_1 | DeepSeek-V4, GLM-5.x, Kimi-K3, Nex-N2.5, Qwen3.8 |
| 5 | tau3-bench | general-tool-usage | tool-use | agent | ✅ | inspect_harbor | sierra_research_tau3_bench | Granite-4.2, Kimi-K3, MiniCPM5, Mistral-M3.5, Nemotron-3.x |
| 5 | HLE w/ tools | knowledge-and-reasoning | expert-knowledge | agent |  | PARTIAL | native hle is no-tools; tool-use variant needs custom solver | DeepSeek-V4, GLM-5.x, Gemma-4, Nemotron-3.x, Qwen3.8 |
| 5 | AutomationBench | general-tool-usage | tool-use | agent |  | MISSING | NOT IMPLEMENTED - top-4 gap | DeepSeek-V4, GLM-5.x, Kimi-K3, Nex-N2.5, Qwen3.8 |
| 5 | Toolathlon | general-tool-usage | tool-use | agent |  | MISSING | NOT IMPLEMENTED - top-4 gap | DeepSeek-V4, GLM-5.x, Kimi-K3, Nex-N2.5, Qwen3.8 |
| 4 | AIME 2025 | knowledge-and-reasoning | math-reasoning | none |  | inspect_evals | aime2025 | Granite-4.2, MiniCPM5, Nemotron-3.x, Step-3.5-Flash |
| 4 | SciCode | agentic-and-coding | research-auto | sandbox |  | inspect_evals | scicode | Granite-4.2, Kimi-K3, MiniCPM5, Nemotron-3.x |
| 4 | tau2-bench | general-tool-usage | tool-use | agent | ✅ | inspect_evals | tau2_airline / tau2_banking / tau2_retail / tau2_telecom | Gemma-4, MiniCPM5, Nemotron-3.x, Step-3.5-Flash |
| 4 | LongBench v2 | long-context | long-context | none |  | PARTIAL | niah / infinite_bench cover long-ctx, not LongBench-v2 | DeepSeek-V4, MiniCPM5, Nemotron-3.x, Qwen3.8 |
| 4 | Agents' Last Exam | general-tool-usage | pro-work | agent |  | MISSING | NOT IMPLEMENTED (harbor ale_rsi_post_training is a different ALE set) | DeepSeek-V4, GLM-5.x, Kimi-K3, Qwen3.8 |
| 4 | IFBench | knowledge-and-reasoning | instruction-following | none |  | MISSING | native ifeval is the older IFEval, not IFBench | Granite-4.2, MiniCPM5, Nemotron-3.x, Qwen3.8 |
| 4 | IMOAnswerBench | knowledge-and-reasoning | math-reasoning | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4, GLM-5.x, Nemotron-3.x, Step-3.5-Flash |
| 4 | OmniDocBench | multimodal | doc-understanding | none |  | MISSING | NOT IMPLEMENTED | Gemma-4, Kimi-K3, Nex-N2.5, Qwen3.8 |
| 3 | AIME 2026 | knowledge-and-reasoning | math-reasoning | none | ✅ | inspect_evals | aime2026 | GLM-5.x, Gemma-4, MiniCPM5 |
| 3 | FrontierSWE | agentic-and-coding | swe-agent | agent |  | inspect_evals | frontier_cs (Frontier-CS, related not same) | GLM-5.x, Kimi-K3, Qwen3.8 |
| 3 | NL2Repo-Bench | agentic-and-coding | swe-agent | agent |  | inspect_harbor | nl2repobench | DeepSeek-V4, GLM-5.x, Qwen3.8 |
| 3 | OSWorld-Verified | general-tool-usage | computer-use | agent |  | inspect_harbor | xlang_ai_osworld_verified  (+ native osworld) | Kimi-K3, Nex-N2.5, Qwen3.8 |
| 3 | ProgramBench | agentic-and-coding | swe-agent | agent |  | inspect_harbor | bencalvert04_programbench | DeepSeek-V4, GLM-5.x, Kimi-K3 |
| 3 | MMMU-Pro | multimodal | vision-reasoning | none |  | PARTIAL | mmmu (base MMMU only, not Pro) | DeepSeek-V4, Gemma-4, Kimi-K3 |
| 3 | MRCR | long-context | long-context | none |  | inspect_evals | mrcr (+ mrcr_2needle/4needle/8needle; neuralmagic/inspect_evals) | DeepSeek-V4, Gemma-4, Qwen3.8 |
| 3 | SWE-bench Multilingual | agentic-and-coding | swe-agent | agent |  | PARTIAL | swe_bench(dataset=...) or harbor abundant_swe_gen_* | DeepSeek-V4, Granite-4.2, Nemotron-3.x |
| 3 | AA-LCR | long-context | long-context | none |  | MISSING | Artificial Analysis proprietary | Kimi-K3, MiniCPM5, Nemotron-3.x |
| 3 | BabyVision | multimodal | vision-perception | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4, Kimi-K3, Qwen3.8 |
| 3 | CritPt | knowledge-and-reasoning | science-reasoning | none |  | MISSING | NOT IMPLEMENTED | GLM-5.x, Kimi-K3, Nemotron-3.x |
| 3 | JobBench | general-tool-usage | pro-work | agent |  | MISSING | NOT IMPLEMENTED | Kimi-K3, Nex-N2.5, Qwen3.8 |
| 3 | MCP-Atlas | general-tool-usage | tool-use | agent |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4, GLM-5.x, Kimi-K3 |
| 3 | MathVision | multimodal | vision-reasoning | none |  | MISSING | native mathvista is different | Gemma-4, Kimi-K3, Qwen3.8 |
| 2 | BBEH | knowledge-and-reasoning | general-reasoning | none |  | inspect_evals | bbeh | DeepSeek-V4, Gemma-4 |
| 2 | BFCL | general-tool-usage | tool-use | agent | ✅ | inspect_evals | bfcl  (+ harbor gorilla_bfcl, 3641 samples) | Granite-4.2, MiniCPM5 |
| 2 | CyberGym | general-tool-usage | cyber | agent |  | inspect_evals | cybergym | DeepSeek-V4, GLM-5.x |
| 2 | GAIA | general-tool-usage | search-research | agent |  | inspect_evals | gaia, gaia_level1/2/3  (+ harbor gaia) | MiniCPM5, Step-3.5-Flash |
| 2 | ZeroBench | multimodal | vision-reasoning | none |  | inspect_evals | zerobench | DeepSeek-V4, Kimi-K3 |
| 2 | Finance Agent | general-tool-usage | pro-work | agent |  | inspect_harbor | vals_financeagent | Kimi-K3, Nemotron-3.x |
| 2 | MMMLU | knowledge-and-reasoning | knowledge-recall | none |  | inspect_harbor | openai_mmmlu | DeepSeek-V4, Gemma-4 |
| 2 | SWE-Marathon | agentic-and-coding | swe-agent | agent |  | inspect_harbor | abundant_swe_marathon | GLM-5.x, Kimi-K3 |
| 2 | BirdBench | general-tool-usage | tool-use | agent |  | PARTIAL | harbor scale_ai_hil_bench carries BIRD-derived text-to-SQL tasks | Granite-4.2, Nemotron-3.x |
| 2 | MLS-Bench-Lite | agentic-and-coding | research-auto | agent |  | PARTIAL | native mle_bench_lite + harbor meta_mlgym_bench (related) | Kimi-K3, Qwen3.8 |
| 2 | Arena-Hard | knowledge-and-reasoning | instruction-following | none |  | MISSING | NOT IMPLEMENTED | Granite-4.2, Nemotron-3.x |
| 2 | BrowseComp-ZH | general-tool-usage | search-research | agent |  | MISSING | native browse_comp is the English set only | MiniCPM5, Step-3.5-Flash |
| 2 | CharXiv | multimodal | vision-reasoning | none |  | MISSING | NOT IMPLEMENTED | Kimi-K3, Qwen3.8 |
| 2 | Codeforces | agentic-and-coding | code-gen | sandbox |  | MISSING | rating, needs live judge | DeepSeek-V4, Gemma-4 |
| 2 | ExploitGym | general-tool-usage | cyber | agent |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4, GLM-5.x |
| 2 | MMLU-ProX | knowledge-and-reasoning | multilingual | none |  | MISSING | NOT IMPLEMENTED | Granite-4.2, Nemotron-3.x |
| 2 | MMLU-Redux | knowledge-and-reasoning | knowledge-recall | none |  | MISSING | native mmlu is base MMLU | DeepSeek-V4, MiniCPM5 |
| 2 | MathArena Apex | knowledge-and-reasoning | math-reasoning | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4, Nemotron-3.x |
| 2 | PostTrainBench | agentic-and-coding | swe-agent | agent |  | MISSING | NOT IMPLEMENTED | GLM-5.x, Kimi-K3 |
| 2 | ProfBench | general-tool-usage | pro-work | agent |  | MISSING | NOT IMPLEMENTED | Granite-4.2, Nemotron-3.x |
| 2 | RULER | long-context | long-context | none |  | MISSING | NOT IMPLEMENTED - niah is the nearest proxy | Granite-4.2, Nemotron-3.x |
| 2 | ResearchRubrics | general-tool-usage | search-research | agent |  | MISSING | NOT IMPLEMENTED | Kimi-K3, Step-3.5-Flash |
| 2 | SuperGPQA | knowledge-and-reasoning | science-reasoning | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4, MiniCPM5 |
| 2 | WebArena-Verified | general-tool-usage | computer-use | agent |  | MISSING | native mind2web is different | Nex-N2.5, Qwen3.8 |
| 1 | AGIEval | knowledge-and-reasoning | knowledge-recall | none |  | inspect_evals | agie_* (9 tasks) | DeepSeek-V4 |
| 1 | BBH | knowledge-and-reasoning | general-reasoning | none |  | inspect_evals | bbh | DeepSeek-V4 |
| 1 | BigCodeBench | agentic-and-coding | code-gen | sandbox |  | inspect_evals | bigcodebench | DeepSeek-V4 |
| 1 | DROP | knowledge-and-reasoning | general-reasoning | none |  | inspect_evals | drop | DeepSeek-V4 |
| 1 | DocVQA | multimodal | doc-understanding | none |  | inspect_evals | docvqa | DeepSeek-V4 |
| 1 | GSM8K | knowledge-and-reasoning | math-reasoning | none |  | inspect_evals | gsm8k | DeepSeek-V4 |
| 1 | HealthBench | knowledge-and-reasoning | domain-professional | none |  | inspect_evals | healthbench | Qwen3.8 |
| 1 | HellaSwag | knowledge-and-reasoning | general-reasoning | none |  | inspect_evals | hellaswag | DeepSeek-V4 |
| 1 | HumanEval | agentic-and-coding | code-gen | sandbox |  | inspect_evals | humaneval | DeepSeek-V4 |
| 1 | IFEval | knowledge-and-reasoning | instruction-following | none |  | inspect_evals | ifeval | MiniCPM5 |
| 1 | LCB-Pro | agentic-and-coding | code-gen | sandbox |  | inspect_evals | livecodebench_pro | MiniCPM5 |
| 1 | MATH | knowledge-and-reasoning | math-reasoning | none |  | inspect_evals | math | DeepSeek-V4 |
| 1 | MGSM | knowledge-and-reasoning | math-reasoning | none |  | inspect_evals | mgsm | DeepSeek-V4 |
| 1 | MMLU | knowledge-and-reasoning | knowledge-recall | none |  | inspect_evals | mmlu_0_shot / mmlu_5_shot | DeepSeek-V4 |
| 1 | PaperBench | agentic-and-coding | research-auto | agent |  | inspect_evals | paperbench | Qwen3.8 |
| 1 | SimpleQA-Verified | knowledge-and-reasoning | factuality | none |  | inspect_evals | simpleqa_verified | DeepSeek-V4 |
| 1 | WinoGrande | knowledge-and-reasoning | general-reasoning | none |  | inspect_evals | winogrande | DeepSeek-V4 |
| 1 | AndroidBench | agentic-and-coding | swe-agent | agent |  | inspect_harbor | android_bench | Qwen3.8 |
| 1 | DeepSearchQA | general-tool-usage | search-research | agent |  | inspect_harbor | kgmon_deepsearchqa | Kimi-K3 |
| 1 | Harvey LAB | general-tool-usage | pro-work | agent |  | inspect_harbor | harveyai_lab | Kimi-K3 |
| 1 | SkillsBench | general-tool-usage | tool-use | agent |  | inspect_harbor | benchflow_skillsbench | Qwen3.8 |
| 1 | MATH-500 | knowledge-and-reasoning | math-reasoning | none |  | PARTIAL | native math is full MATH; the 500-subset is not a separate task | MiniCPM5 |
| 1 | OSWorld-G | general-tool-usage | computer-use | agent |  | PARTIAL | grounding subset; native osworld / harbor osworld-verified are the main set | Nex-N2.5 |
| 1 | AA-Briefcase | general-tool-usage | pro-work | agent |  | MISSING | Artificial Analysis proprietary | Kimi-K3 |
| 1 | AA-Omniscience | knowledge-and-reasoning | knowledge-recall | none |  | MISSING | Artificial Analysis proprietary | Nemotron-3.x |
| 1 | APEX-Agents | general-tool-usage | pro-work | agent |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | AndroidWorld | general-tool-usage | computer-use | agent |  | MISSING | harbor android_bench is a different set | Qwen3.8 |
| 1 | C-Eval | knowledge-and-reasoning | knowledge-recall | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | CLUEWSC | knowledge-and-reasoning | general-reasoning | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | CMMLU | knowledge-and-reasoning | knowledge-recall | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | CMath | knowledge-and-reasoning | math-reasoning | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | CVBench | multimodal | vision-perception | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | Chartography | multimodal | vision-reasoning | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | Chinese-SimpleQA | knowledge-and-reasoning | factuality | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | Claw-Gym | general-tool-usage | tool-use | agent |  | MISSING | NOT IMPLEMENTED | MiniCPM5 |
| 1 | CoVoST | multimodal | audio | none |  | MISSING | NOT IMPLEMENTED - neither package has audio evals | Gemma-4 |
| 1 | CoWorkBench | general-tool-usage | pro-work | agent |  | MISSING | in-house/unreleased | Qwen3.8 |
| 1 | CorpFin v2 | general-tool-usage | pro-work | agent |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | CorpusQA 1M | long-context | long-context | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | DSBench | general-tool-usage | research-auto | agent |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | ERQA | multimodal | vision-perception | none |  | MISSING | NOT IMPLEMENTED | Qwen3.8 |
| 1 | ExploitBench | general-tool-usage | cyber | agent |  | MISSING | NOT IMPLEMENTED | GLM-5.x |
| 1 | FACTS Parametric | knowledge-and-reasoning | factuality | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | FLEURS | multimodal | audio | none |  | MISSING | NOT IMPLEMENTED - neither package has audio evals | Gemma-4 |
| 1 | IOI 2025 | agentic-and-coding | code-gen | sandbox |  | MISSING | NOT IMPLEMENTED - harbor usaco is a different olympiad set | Nemotron-3.x |
| 1 | Legal Research Bench | general-tool-usage | pro-work | agent |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | LongBenchPro | long-context | long-context | none |  | MISSING | NOT IMPLEMENTED | MiniCPM5 |
| 1 | MCPMark | general-tool-usage | tool-use | agent |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | MMVU | multimodal | vision-reasoning | none |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | MedXPertQA | multimodal | vision-reasoning | none |  | MISSING | NOT IMPLEMENTED | Gemma-4 |
| 1 | Multi-Challenge | knowledge-and-reasoning | instruction-following | none |  | MISSING | NOT IMPLEMENTED | Nemotron-3.x |
| 1 | Multi-IF | knowledge-and-reasoning | instruction-following | none |  | MISSING | NOT IMPLEMENTED | MiniCPM5 |
| 1 | MultiLoKo | knowledge-and-reasoning | multilingual | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | NoLiMa | long-context | long-context | none |  | MISSING | NOT IMPLEMENTED | MiniCPM5 |
| 1 | OJBench | agentic-and-coding | code-gen | sandbox |  | MISSING | NOT IMPLEMENTED | MiniCPM5 |
| 1 | OfficeQA Pro | general-tool-usage | pro-work | agent |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | PerceptionBench | multimodal | vision-perception | none |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | PinchBench | general-tool-usage | tool-use | agent |  | MISSING | NOT IMPLEMENTED | Nemotron-3.x |
| 1 | QwenClaw | general-tool-usage | tool-use | agent |  | MISSING | NOT IMPLEMENTED | MiniCPM5 |
| 1 | RealWorldQA | multimodal | vision-perception | none |  | MISSING | NOT IMPLEMENTED | Qwen3.8 |
| 1 | RefCOCO | multimodal | vision-perception | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | SEC-Bench Pro | general-tool-usage | cyber | agent |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | SWE-MM | agentic-and-coding | swe-agent | agent |  | MISSING | NOT IMPLEMENTED | Nex-N2.5 |
| 1 | SaaS-Bench | general-tool-usage | pro-work | agent |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | SpreadsheetBench 2 | general-tool-usage | pro-work | agent |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | Terminal-Bench 3.0 | agentic-and-coding | swe-agent | agent |  | MISSING | not yet packaged (2.0/2.1/Pro/Science are) | GLM-5.x |
| 1 | TriviaQA | knowledge-and-reasoning | knowledge-recall | none |  | MISSING | NOT IMPLEMENTED | DeepSeek-V4 |
| 1 | Video-MME | multimodal | vision-reasoning | none |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | Vision2Web | multimodal | vision-reasoning | none |  | MISSING | NOT IMPLEMENTED | Nex-N2.5 |
| 1 | WMT24++ | knowledge-and-reasoning | multilingual | none |  | MISSING | NOT IMPLEMENTED | Nemotron-3.x |
| 1 | WebTest | general-tool-usage | computer-use | agent |  | MISSING | NOT IMPLEMENTED | Nex-N2.5 |
| 1 | WideSearch | general-tool-usage | search-research | agent |  | MISSING | NOT IMPLEMENTED | Qwen3.8 |
| 1 | WildClaw | general-tool-usage | tool-use | agent |  | MISSING | NOT IMPLEMENTED | MiniCPM5 |
| 1 | WorkSpaceBench | general-tool-usage | pro-work | agent |  | MISSING | in-house/unreleased | Qwen3.8 |
| 1 | WorldVQA | multimodal | vision-reasoning | none |  | MISSING | NOT IMPLEMENTED | Kimi-K3 |
| 1 | xbench-DeepSearch | general-tool-usage | search-research | agent |  | MISSING | NOT IMPLEMENTED | 61, runnable=27 (44%), partial=7, missing=27 |

## Coverage by bucket

| | total | runnable | partial | missing | % |
|---|---|---|---|---|---|
| agentic-and-coding | 23 | 15 | 2 | 6 | 65% |
| general-tool-usage | 45 | 12 | 2 | 31 | 26% |
| knowledge-and-reasoning | 42 | 19 | 2 | 21 | 45% |
| long-context | 7 | 0 | 2 | 5 | 0% |
| multimodal | 20 | 2 | 1 | 17 | 10% |

- **agentic-and-coding** software engineering and code generation, agentic or single-shot
- **general-tool-usage** non-coding agent work: tool use, computer use, search, professional tasks, cyber
- **knowledge-and-reasoning** knowledge, reasoning, math, factuality, instruction following
- **long-context** retrieval and reasoning over very long inputs
- **multimodal** vision, document, chart and video understanding

## Coverage by execution tier

| | total | runnable | partial | missing | % |
|---|---|---|---|---|---|
| none | 68 | 21 | 4 | 43 | 30% |
| sandbox | 8 | 5 | 0 | 3 | 62% |
| agent | 61 | 22 | 5 | 34 | 36% |

- **none** prompt in, answer out; string-match or LLM-judge scoring. No Docker.
- **sandbox** single-shot generation, but code must run to be scored. Sandbox, no agent loop.
- **agent** multi-turn agent loop in a stateful environment. Docker, tools, high variance.

## Cross-tab: bucket x exec tier (runnable/total)

| | none | sandbox | agent | total |
|---|---|---|---|---|
| agentic-and-coding | - | 5/8 | 10/15 | 15/23 |
| general-tool-usage | - | - | 12/45 | 12/45 |
| knowledge-and-reasoning | 19/41 | - | 0/1 | 19/42 |
| long-context | 0/7 | - | - | 0/7 |
| multimodal | 2/20 | - | - | 2/20 |

## Detail by bucket and subclass

Status: ✔ runnable, ~ partial (substitute only), ✗ missing. `#` = families reporting. **Repo** = implemented here.

### AGENTIC-AND-CODING  (15/23 runnable) — software engineering and code generation, agentic or single-shot

#### swe-agent (9/13 runnable) — long-horizon software engineering in a sandbox/container

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 9 | Terminal-Bench 2.1 | agent | terminal_bench_2_1 | ✅ |
| ✔ | 6 | SWE-bench Verified | agent | swe_bench | ✅ |
| ✔ | 6 | SWE-bench Pro | agent | scale_ai_swe_bench_pro / cais_swebenchpro | ✅ |
| ✔ | 5 | DeepSWE 1.1 | agent | datacurve_deep_swe_1_1 | ✅ |
| ✔ | 3 | FrontierSWE | agent | frontier_cs (Frontier-CS, related not same) |  |
| ✔ | 3 | NL2Repo-Bench | agent | nl2repobench |  |
| ✔ | 3 | ProgramBench | agent | bencalvert04_programbench |  |
| ~ | 3 | SWE-bench Multilingual | agent |  |  |
| ✔ | 2 | SWE-Marathon | agent | abundant_swe_marathon |  |
| ✗ | 2 | PostTrainBench | agent |  |  |
| ✔ | 1 | AndroidBench | agent | android_bench |  |
| ✗ | 1 | SWE-MM | agent |  |  |
| ✗ | 1 | Terminal-Bench 3.0 | agent |  |  |

#### code-gen (4/7 runnable) — single-shot code generation, no environment

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 7 | LiveCodeBench | sandbox | livecodebench  (+ native livecodebench_pro = different bench) | ✅ |
| ✗ | 2 | Codeforces | sandbox |  |  |
| ✔ | 1 | BigCodeBench | sandbox | bigcodebench |  |
| ✔ | 1 | HumanEval | sandbox | humaneval |  |
| ✔ | 1 | LCB-Pro | sandbox | livecodebench_pro |  |
| ✗ | 1 | IOI 2025 | sandbox |  |  |
| ✗ | 1 | OJBench | sandbox |  |  |

#### research-auto (2/3 runnable) — ML/science research replication and data-science pipelines

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 4 | SciCode | sandbox | scicode |  |
| ~ | 2 | MLS-Bench-Lite | agent |  |  |
| ✔ | 1 | PaperBench | agent | paperbench |  |

### GENERAL-TOOL-USAGE  (12/45 runnable) — non-coding agent work: tool use, computer use, search, professional tasks, cyber

#### pro-work (3/15 runnable) — economically-valuable professional work (O*NET-style occupational tasks)

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 7 | GDPval-AA | agent | gdpval |  |
| ✗ | 4 | Agents' Last Exam | agent |  |  |
| ✗ | 3 | JobBench | agent |  |  |
| ✔ | 2 | Finance Agent | agent | vals_financeagent |  |
| ✗ | 2 | ProfBench | agent |  |  |
| ✔ | 1 | Harvey LAB | agent | harveyai_lab |  |
| ✗ | 1 | AA-Briefcase | agent |  |  |
| ✗ | 1 | APEX-Agents | agent |  |  |
| ✗ | 1 | CoWorkBench | agent |  |  |
| ✗ | 1 | CorpFin v2 | agent |  |  |
| ✗ | 1 | Legal Research Bench | agent |  |  |
| ✗ | 1 | OfficeQA Pro | agent |  |  |
| ✗ | 1 | SaaS-Bench | agent |  |  |
| ✗ | 1 | SpreadsheetBench 2 | agent |  |  |
| ✗ | 1 | WorkSpaceBench | agent |  |  |

#### tool-use (4/13 runnable) — multi-app / MCP tool orchestration, execution-verified end state

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 5 | tau3-bench | agent | sierra_research_tau3_bench | ✅ |
| ✗ | 5 | AutomationBench | agent |  |  |
| ✗ | 5 | Toolathlon | agent |  |  |
| ✔ | 4 | tau2-bench | agent | tau2_airline / tau2_banking / tau2_retail / tau2_telecom | ✅ |
| ✗ | 3 | MCP-Atlas | agent |  |  |
| ✔ | 2 | BFCL | agent | bfcl  (+ harbor gorilla_bfcl, 3641 samples) | ✅ |
| ~ | 2 | BirdBench | agent |  |  |
| ✔ | 1 | SkillsBench | agent | benchflow_skillsbench |  |
| ✗ | 1 | Claw-Gym | agent |  |  |
| ✗ | 1 | MCPMark | agent |  |  |
| ✗ | 1 | PinchBench | agent |  |  |
| ✗ | 1 | QwenClaw | agent |  |  |
| ✗ | 1 | WildClaw | agent |  |  |

#### search-research (3/7 runnable) — web browsing and deep research

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 6 | BrowseComp | agent | browse_comp |  |
| ✔ | 2 | GAIA | agent | gaia, gaia_level1/2/3  (+ harbor gaia) |  |
| ✗ | 2 | BrowseComp-ZH | agent |  |  |
| ✗ | 2 | ResearchRubrics | agent |  |  |
| ✔ | 1 | DeepSearchQA | agent | kgmon_deepsearchqa |  |
| ✗ | 1 | WideSearch | agent |  |  |
| ✗ | 1 | xbench-DeepSearch | agent |  |  |

#### computer-use (1/5 runnable) — GUI, desktop, browser or mobile control

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 3 | OSWorld-Verified | agent | xlang_ai_osworld_verified  (+ native osworld) |  |
| ✗ | 2 | WebArena-Verified | agent |  |  |
| ~ | 1 | OSWorld-G | agent |  |  |
| ✗ | 1 | AndroidWorld | agent |  |  |
| ✗ | 1 | WebTest | agent |  |  |

#### cyber (1/4 runnable) — vulnerability discovery and exploitation

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 2 | CyberGym | agent | cybergym |  |
| ✗ | 2 | ExploitGym | agent |  |  |
| ✗ | 1 | ExploitBench | agent |  |  |
| ✗ | 1 | SEC-Bench Pro | agent |  |  |

#### research-auto (0/1 runnable) — ML/science research replication and data-science pipelines

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✗ | 1 | DSBench | agent |  |  |

### KNOWLEDGE-AND-REASONING  (19/42 runnable) — knowledge, reasoning, math, factuality, instruction following

#### math-reasoning (5/10 runnable) — mathematical problem solving

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✗ | 6 | HMMT | none |  |  |
| ✔ | 4 | AIME 2025 | none | aime2025 |  |
| ✗ | 4 | IMOAnswerBench | none |  |  |
| ✔ | 3 | AIME 2026 | none | aime2026 | ✅ |
| ✗ | 2 | MathArena Apex | none |  |  |
| ✔ | 1 | GSM8K | none | gsm8k |  |
| ✔ | 1 | MATH | none | math |  |
| ✔ | 1 | MGSM | none | mgsm |  |
| ~ | 1 | MATH-500 | none |  |  |
| ✗ | 1 | CMath | none |  |  |

#### knowledge-recall (4/9 runnable) — academic/world knowledge recall, mostly multiple-choice

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 5 | MMLU-Pro | none | mmlu_pro | ✅ |
| ✔ | 2 | MMMLU | none | openai_mmmlu |  |
| ✗ | 2 | MMLU-Redux | none |  |  |
| ✔ | 1 | AGIEval | none | agie_* (9 tasks) |  |
| ✔ | 1 | MMLU | none | mmlu_0_shot / mmlu_5_shot |  |
| ✗ | 1 | AA-Omniscience | none |  |  |
| ✗ | 1 | C-Eval | none |  |  |
| ✗ | 1 | CMMLU | none |  |  |
| ✗ | 1 | TriviaQA | none |  |  |

#### general-reasoning (5/6 runnable) — logic, commonsense, reading comprehension

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 2 | BBEH | none | bbeh |  |
| ✔ | 1 | BBH | none | bbh |  |
| ✔ | 1 | DROP | none | drop |  |
| ✔ | 1 | HellaSwag | none | hellaswag |  |
| ✔ | 1 | WinoGrande | none | winogrande |  |
| ✗ | 1 | CLUEWSC | none |  |  |

#### instruction-following (1/5 runnable) — adherence to explicit output constraints

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✗ | 4 | IFBench | none |  |  |
| ✗ | 2 | Arena-Hard | none |  |  |
| ✔ | 1 | IFEval | none | ifeval |  |
| ✗ | 1 | Multi-Challenge | none |  |  |
| ✗ | 1 | Multi-IF | none |  |  |

#### factuality (1/3 runnable) — short-fact accuracy and hallucination resistance

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 1 | SimpleQA-Verified | none | simpleqa_verified |  |
| ✗ | 1 | Chinese-SimpleQA | none |  |  |
| ✗ | 1 | FACTS Parametric | none |  |  |

#### multilingual (0/3 runnable) — cross-lingual capability

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✗ | 2 | MMLU-ProX | none |  |  |
| ✗ | 1 | MultiLoKo | none |  |  |
| ✗ | 1 | WMT24++ | none |  |  |

#### science-reasoning (1/3 runnable) — graduate-to-frontier STEM reasoning

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 8 | GPQA Diamond | none | gpqa_diamond | ✅ |
| ✗ | 3 | CritPt | none |  |  |
| ✗ | 2 | SuperGPQA | none |  |  |

#### expert-knowledge (1/2 runnable) — multidisciplinary expert-level closed-ended questions

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 7 | HLE | none | hle |  |
| ~ | 5 | HLE w/ tools | agent |  |  |

#### domain-professional (1/1 runnable) — professional domain knowledge (health/legal/finance), non-agentic

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✔ | 1 | HealthBench | none | healthbench |  |

### LONG-CONTEXT  (1/7 runnable) — retrieval and reasoning over very long inputs

#### long-context (1/7 runnable) — retrieval and reasoning over very long inputs

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ~ | 4 | LongBench v2 | none |  |  |
| ✔ | 3 | MRCR | none | mrcr (+ mrcr_2needle / mrcr_4needle / mrcr_8needle) |  |
| ✗ | 3 | AA-LCR | none |  |  |
| ✗ | 2 | RULER | none |  |  |
| ✗ | 1 | CorpusQA 1M | none |  |  |
| ✗ | 1 | LongBenchPro | none |  |  |
| ✗ | 1 | NoLiMa | none |  |  |

### MULTIMODAL  (2/20 runnable) — vision, document, chart and video understanding

#### vision-reasoning (1/10 runnable) — multimodal reasoning over images, charts and video

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ~ | 3 | MMMU-Pro | none |  |  |
| ✗ | 3 | MathVision | none |  |  |
| ✔ | 2 | ZeroBench | none | zerobench |  |
| ✗ | 2 | CharXiv | none |  |  |
| ✗ | 1 | Chartography | none |  |  |
| ✗ | 1 | MMVU | none |  |  |
| ✗ | 1 | MedXPertQA | none |  |  |
| ✗ | 1 | Video-MME | none |  |  |
| ✗ | 1 | Vision2Web | none |  |  |
| ✗ | 1 | WorldVQA | none |  |  |

#### vision-perception (0/6 runnable) — low-level visual perception, spatial and embodied grounding

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✗ | 3 | BabyVision | none |  |  |
| ✗ | 1 | CVBench | none |  |  |
| ✗ | 1 | ERQA | none |  |  |
| ✗ | 1 | PerceptionBench | none |  |  |
| ✗ | 1 | RealWorldQA | none |  |  |
| ✗ | 1 | RefCOCO | none |  |  |

#### audio (0/2 runnable) — speech recognition and speech translation

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✗ | 1 | CoVoST | none |  |  |
| ✗ | 1 | FLEURS | none |  |  |

#### doc-understanding (1/2 runnable) — document and OCR understanding

| status | # | Benchmark | Exec | Registry task / note | Repo |
|---|---|---|---|---|---|
| ✗ | 4 | OmniDocBench | none |  |  |
| ✔ | 1 | DocVQA | none | docvqa |  |

## Legend

- **#**: number of model families (of 11 surveyed) reporting this benchmark
- **Evaluates**: capability bucket, aligned to model-card section headers:
  - `agentic-and-coding`: software engineering and code generation, agentic or single-shot
  - `general-tool-usage`: non-coding agent work: tool use, computer use, search, professional tasks, cyber
  - `knowledge-and-reasoning`: knowledge, reasoning, math, factuality, instruction following
  - `long-context`: retrieval and reasoning over very long inputs
  - `multimodal`: vision, document, chart and video understanding
- **Subclass**: fine-grained capability. The 'agentic-' prefix was removed deliberately:
  agentic-ness is an Exec property, not a capability. DeepSWE = coding +
  swe-agent + agent. Terminal-Bench, SWE-bench and DeepSWE are CODING
  because that is where Kimi/Qwen/DeepSeek file them on their own cards.
- **Exec**: infrastructure required. Orthogonal to capability:
  - `none`: prompt in, answer out; string-match or LLM-judge scoring. No Docker.
  - `sandbox`: single-shot generation, but code must run to be scored. Sandbox, no agent loop.
  - `agent`: multi-turn agent loop in a stateful environment. Docker, tools, high variance.
  For MISSING benchmarks the Exec tier is inferred from the paper, not
  verified against an implementation.
- **Where**: inspect_evals | inspect_harbor | PARTIAL (substitute only) | MISSING
- **Repo**: ✅ = implemented and run in this repository (for at least one model); results and protocol are in
  `<org>_<model>/<category>/<task>/README.md`.
- **Detail**: registry task name when runnable, else an explanatory note
