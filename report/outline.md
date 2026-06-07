# ACL-Style Report Outline

This is a paragraph-level outline for a 6-8 page ACL-style course project report. It is not final prose.

## Abstract

### Main Claim

- EmoFlow is an appraisal-grounded, Bayesian-inspired classifier for conversational emotion recognition that separates utterance evidence from dialogue context through interpretable appraisal vectors, exponential-decay memory, and log-additive prior/likelihood fusion.

### Supporting Source Files

- `REPORT.md.txt`
- `README.md.txt`
- `EmoFlow_Presentation.pptx` Office XML text
- `model.py.txt`
- `encoder.py.txt`
- `memory.py.txt`
- `bayes.py.txt`

### Details for Final Paper

- One sentence on ERC and MELD.
- One sentence on the three-component method.
- One sentence on the central implementation/debugging lesson: removing sigmoid from the appraisal bottleneck fixed representation collapse.
- One sentence on controlled results: EmoFlow learned lambda wF1 0.6171, no-decay 0.6052, stateless 0.5631, LSTM 0.4241 under matched MELD+DD-rare3 setup.
- One caveat sentence: headline depends on selective DailyDialog augmentation and is not a controlled superiority claim over published MELD baselines.

### Missing or Uncertain Information

- Exact citation metadata for all related work must be verified.
- Checkpoint files are not included, so result claims trace to REPORT/README/slides rather than raw metrics files.

## Introduction

### Main Claim

- ERC requires modeling utterance content, dialogue context, temporal emotion persistence, and severe class imbalance; EmoFlow addresses these with appraisal-grounded representations and structured temporal inference.

### Supporting Source Files

- `REPORT.md.txt`
- `README.md.txt`
- `EmoFlow_Presentation.pptx` Office XML text
- `dataloader.py.txt`
- `evaluation.py.txt`

### Details for Final Paper

- Paragraph 1: Define ERC as per-turn emotion classification over dialogues; specify the seven-label MELD vocabulary.
- Paragraph 2: Explain the three project-specific challenges: context dependence, emotional inertia, and long-tailed labels.
- Paragraph 3: Introduce the gap: existing ERC models can model context but often entangle current text, speaker/context, and emotion state in opaque hidden states.
- Paragraph 4: Present EmoFlow's core idea: use Scherer appraisal dimensions as an interpretable bottleneck, then combine current appraisal evidence and temporal prior through Bayesian-style fusion.
- Paragraph 5: State contributions conservatively: implemented architecture, controlled in-repo ablations, sigmoid-saturation diagnosis, and selective rare-class augmentation.
- Paragraph 6: State scope/deviation from the proposal: response generation and human coherence evaluation were planned but deferred; final paper focuses on the classifier.

### Missing or Uncertain Information

- Published related-work comparison must avoid broad SOTA claims unless citations and training settings are verified.
- Proposal details should be labeled planned/proposed unless supported by REPORT/slides or code.
- Exact MELD dataset split statistics should be used only if supported by REPORT/README or extracted data files.

## Method

### Main Claim

- EmoFlow factorizes ERC into interpretable appraisal extraction, structured temporal memory, and log-additive prior/likelihood fusion.

### Supporting Source Files

- `REPORT.md.txt`
- `README.md.txt`
- `emoflow_algorithm_explained.md.txt`
- `EmoFlow_Presentation.pptx` Office XML text
- `model.py.txt`
- `encoder.py.txt`
- `memory.py.txt`
- `bayes.py.txt`
- `appraisal_targets.py.txt`
- `dataloader.py.txt`
- `train.py.txt`

### Details for Final Paper

- Paragraph 1: Define the problem setup: a dialogue of utterances and per-turn labels, with model outputs for six non-neutral classes plus neutral thresholding in multilabel mode.
- Paragraph 2: Describe appraisal target construction from Scherer Table 5.5: 8 dimensions, min-max normalized prototypes, derived surprise vector, neutral/masked labels.
- Paragraph 3: Describe StimulusEncoder: frozen backbone, LLaMA-3-8B final target, LoRA adapters, QLoRA for LLaMA, last-token pooling, two-layer AppraisalHead, no sigmoid.
- Paragraph 4: Describe TemporalMemory: causal exponential-decay softmax over past appraisals, lambda as a learnable softplus scalar, no-decay ablation at lambda near zero.
- Paragraph 5: Describe BayesianHead: prior MLP from memory, likelihood MLP from current appraisal, posterior logits as their sum.
- Paragraph 6: Describe output/training formulation: multilabel BCE over six non-neutral emotions, neutral as all-zero target, sigmoid threshold selected on dev.
- Paragraph 7: Describe joint loss: emotion loss plus appraisal MSE weighted by `appraisal_alpha`; mention alpha 0.1 for final reported configuration if retained.
- Paragraph 8: Describe deviations from proposal: 8 dimensions rather than 5, log-additive MLP rather than explicit Dirichlet-Categorical update, classifier-only delivery.

### Missing or Uncertain Information

- `dataloader.py` docstring still mentions a 5-dimensional target in one place, while implementation and `appraisal_targets.py` use 8 dimensions; flag as code-comment inconsistency, not method uncertainty.
- Exact total/trainable parameter counts should be checked against the final run or kept approximate.
- The "mathematically equivalent" Dirichlet-Categorical claim should be softened unless a formal derivation is included.

## Experiments

### Main Claim

- Under matched data and training conditions, EmoFlow's structured memory outperforms stateless and LSTM alternatives, while results also reveal limits from augmentation dependence, rare-class failure, and incomplete planned evaluations.

### Supporting Source Files

- `REPORT.md.txt`
- `README.md.txt`
- `EmoFlow_Presentation.pptx` Office XML text
- `train.py.txt`
- `train_appraisal.py.txt`
- `dataloader.py.txt`
- `baselines.py.txt`
- `evaluation.py.txt`
- `make_table.py.txt`

### Details for Final Paper

- Paragraph 1: Experimental setup: MELD primary benchmark, DailyDialog rare-class augmentation, text-only setting, seven-class evaluation, no shipped raw datasets.
- Paragraph 2: Data handling: preprocessing into dialogue JSONL, label normalization, dialogue-level batching, selective DailyDialog dialogues containing fear/disgust/sadness, dialogue-level oversampling.
- Paragraph 3: Training setup: LLaMA-3-8B with LoRA, multilabel BCE, appraisal MSE alpha 0.1, AdamW, learning rate 5e-4, batch size 2, three epochs, seed 42, threshold sweep selecting tau 0.2.
- Paragraph 4: Metrics: weighted F1, weighted F1 over non-neutral turns, and Emotion Transition Accuracy.
- Paragraph 5: Main controlled table: EmoFlow learned lambda, EmoFlow lambda=0, stateless, and LSTM under identical MELD+DD-rare3+oversample conditions. Include only numbers from REPORT/README/slides.
- Paragraph 6: Failure analysis: six mitigation attempts failed before diagnosing sigmoid saturation; bounded bottleneck caused constant appraisal vectors and blocked useful gradients.
- Paragraph 7: Fix and impact: removing sigmoid made appraisal outputs input-dependent and enabled later imbalance strategies; present numbers only as reported, with caveat that raw checkpoint evidence is not included.
- Paragraph 8: Related benchmark context: compare to published MELD numbers only as contextual and caveated because EmoFlow uses DailyDialog augmentation.
- Paragraph 9: Limitations: no final EmoryNLP/IEMOCAP runs, no response generator, no human coherence score, fear class failure, text-only MELD, approximate Bayesian update.

### Missing or Uncertain Information

- Raw checkpoint metrics are not present because `ckpt/` and `ckpt_lambda/` are skipped/gitignored.
- Exact per-class precision/recall/F1 table should be copied only from REPORT/slides if used.
- Final EmoryNLP results after the sigmoid fix are missing.
- IEMOCAP was not attempted.
- Human coherence evaluation was proposed but not performed.
- Appraisal-only pretraining is implemented but REPORT says it was not used in the final pipeline.

## Related Work

### Main Claim

- EmoFlow connects ERC models, appraisal-theoretic emotion modeling, probabilistic/Bayesian factorization, temporal emotion dynamics, and parameter-efficient LLM adaptation.

### Supporting Source Files

- `REPORT.md.txt`
- `README.md.txt`
- `EmoFlow_Presentation.pptx` Office XML text
- `appraisal_targets.py.txt`
- `Paper_*.txt` filenames only, because extracted texts are empty
- `ECS_271_ML_-_Project_Proposal_acl_latex.tex.txt` for course requirement context

### Details for Final Paper

- Paragraph 1: Conversational emotion recognition: MELD, DialogueRNN, DialogueGCN/COSMIC/MMGCN if citations are verified from REPORT or papers.
- Paragraph 2: Appraisal theory: Scherer's Component Process Model and ISEAR appraisal profiles, focusing on why appraisals provide an interpretable bottleneck.
- Paragraph 3: Temporal and context modeling: contrast generic recurrent/graph context modeling with EmoFlow's single-scalar exponential decay.
- Paragraph 4: Probabilistic/Bayesian emotion modeling: explain EmoFlow's Bayesian-inspired prior/likelihood factorization; avoid claiming full Bayesian inference.
- Paragraph 5: Neural and LLM-based emotion modeling: cite LoRA for parameter-efficient adaptation and any verified LLM emotion papers from `Paper/`.
- Paragraph 6: Multi-label or rare-class emotion prediction: discuss the neutral-as-all-zero multilabel reformulation if supported by references or frame as a project design choice rather than a literature claim.

### Missing or Uncertain Information

- The `Paper_*.txt` extracted files are empty, so their technical claims cannot yet be summarized.
- Citation keys and full bibliography entries need to be verified from the original PDFs or reliable metadata.
- Do not invent arguments for reference papers from filenames alone.

## Conclusion and Future Work

### Main Claim

- EmoFlow demonstrates that appraisal-grounded structure and simple temporal priors can provide useful inductive bias for ERC, while the final system remains limited by incomplete generation evaluation, missing multi-dataset validation, rare-class weakness, and approximate Bayesian inference.

### Supporting Source Files

- `REPORT.md.txt`
- `README.md.txt`
- `EmoFlow_Presentation.pptx` Office XML text
- `evaluation.py.txt`
- `train_appraisal.py.txt`

### Details for Final Paper

- Paragraph 1: Summarize delivered contributions: classifier architecture, appraisal bottleneck, exponential memory, BayesianHead, and sigmoid-saturation diagnosis.
- Paragraph 2: Summarize conservative findings: controlled matched-data ablations favor EmoFlow over stateless and LSTM baselines; headline 0.62 depends on DailyDialog rare-class augmentation.
- Paragraph 3: Summarize limitations: no post-fix EmoryNLP/IEMOCAP, no response generator, no coherence evaluation, no multimodal MELD, fear class remains unresolved.
- Paragraph 4: Future work: implement response generator and coherence protocol, evaluate post-fix pipeline on EmoryNLP/IEMOCAP, add multimodal inputs, target rare-class failure, explore larger LoRA/partial unfreezing, and revisit explicit Bayesian inference.

### Missing or Uncertain Information

- Future work should not imply these components are implemented.
- Claims about transfer of the sigmoid-saturation diagnosis to other datasets should be phrased as plausible future evaluation, not established fact.
