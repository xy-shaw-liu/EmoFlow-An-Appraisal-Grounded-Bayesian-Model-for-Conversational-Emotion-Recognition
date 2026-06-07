# Claim Risk Report

## High-Risk Claims

### "EmoFlow is fully Bayesian."

- Risk: The implementation uses two MLPs and logit addition, not explicit Bayesian posterior inference or a Dirichlet-Categorical update.
- Evidence: `report/extracted_sources/bayes.py.txt`; `report/extracted_sources/REPORT.md.txt`.
- Safer rewrite: "EmoFlow uses a Bayesian-inspired prior-likelihood fusion implemented as log-additive neural logits."

### "EmoFlow outperforms published ERC baselines."

- Risk: REPORT.md itself says the comparison is indicative, not controlled, because EmoFlow uses selective DailyDialog augmentation while published baselines may train only on MELD.
- Evidence: `report/extracted_sources/REPORT.md.txt`; `report/related_work_plan.md`.
- Safer rewrite: "Under our in-repository matched setup, EmoFlow outperforms the implemented stateless and LSTM baselines. Published MELD numbers provide context but are not a controlled comparison."

### "The method is state-of-the-art."

- Risk: Unsupported and contradicted by project caveats; COSMIC is reported above EmoFlow in the contextual comparison, and training settings differ.
- Evidence: `report/extracted_sources/REPORT.md.txt`.
- Safer rewrite: Do not claim state-of-the-art performance. Use: "EmoFlow reaches competitive performance in the project report's MELD setting, with important augmentation and reproducibility caveats."

### "The appraisal vectors are ground-truth psychological annotations."

- Risk: Appraisal vectors are derived from emotion-level Scherer/ISEAR prototypes and assigned by label; they are not human annotations for each utterance.
- Evidence: `report/extracted_sources/appraisal_targets.py.txt`; `report/extracted_sources/dataloader.py.txt`.
- Safer rewrite: "The appraisal vectors are weakly supervised targets derived from emotion-level appraisal prototypes."

### "Response generation was completed."

- Risk: REPORT.md, slides, and evaluation code state response generation and Coherence Score were deferred/not implemented.
- Evidence: `report/extracted_sources/REPORT.md.txt`; `report/extracted_sources/EmoFlow_Presentation.pptx.txt`; `report/extracted_sources/evaluation.py.txt`.
- Safer rewrite: "Response generation was proposed but deferred; this report focuses on classification."

### "All experiments are fully reproducible from the repository."

- Risk: Raw datasets, processed data, checkpoints, and final metrics files are excluded; results are documented in reports but not independently verifiable from included artifacts alone.
- Evidence: `report/extracted_sources/README.md.txt`; `report/source_inventory.md`; `report/extracted_sources/make_table.py.txt`.
- Safer rewrite: "Reported results are documented in project reports, but checkpoints, processed data, and raw datasets are not fully included."

### "The headline 0.62 wF1 demonstrates architecture-only improvement."

- Risk: REPORT.md says the 0.62 headline depends heavily on selective DailyDialog rare-class augmentation; MELD-only post-fix score is reported as 0.42.
- Evidence: `report/extracted_sources/REPORT.md.txt`; `report/extracted_sources/README.md.txt`.
- Safer rewrite: "The headline 0.62 wF1 is obtained with selective DailyDialog rare-class augmentation; architecture effects are better assessed through the matched in-repository ablations."

## Medium-Risk Claims

### "Removing sigmoid fixed the model."

- Risk: Supported by project report, slides, and encoder comments, but raw diagnostic logs/checkpoints are not included.
- Evidence: `report/extracted_sources/REPORT.md.txt`; `report/extracted_sources/encoder.py.txt`; `report/extracted_sources/EmoFlow_Presentation.pptx.txt`.
- Safer rewrite: "The project report traces a representation-collapse failure to sigmoid saturation and reports that removing the sigmoid restored input-dependent appraisals."

### "The final reported hyperparameters are exact."

- Risk: README and slides report commands/settings, but final checkpoint `args.json` is not included.
- Evidence: `report/extracted_sources/README.md.txt`; `report/extracted_sources/EmoFlow_Presentation.pptx.txt`; `report/source_inventory.md`.
- Safer rewrite: "The reported run configuration uses..."

### "The model has ~20M trainable parameters."

- Risk: Supported by report/README/slides but exact count depends on final backbone and run configuration.
- Evidence: `report/extracted_sources/README.md.txt`; `report/extracted_sources/EmoFlow_Presentation.pptx.txt`; `report/extracted_sources/encoder.py.txt`.
- Safer rewrite: "The reported LLaMA-LoRA configuration trains approximately 20M parameters."

### "The learned lambda memory is generally better than LSTM memory."

- Risk: The local LSTM is a BiLSTM-over-appraisals baseline, not DialogueRNN or all recurrent models.
- Evidence: `report/extracted_sources/baselines.py.txt`; `report/extracted_sources/REPORT.md.txt`.
- Safer rewrite: "In the matched project ablation, exponential-decay memory outperforms the implemented BiLSTM-over-appraisals baseline."

### "MELD-only post-fix score is 0.42."

- Risk: Reported in project documents but not backed by included checkpoint metrics.
- Evidence: `report/extracted_sources/REPORT.md.txt`; `report/extracted_sources/EmoFlow_Presentation.pptx.txt`.
- Safer rewrite: "The project report documents a MELD-only post-fix wF1 of 0.42."

### "Fear F1 is 0."

- Risk: Reported in project documents/slides; raw predictions are not included.
- Evidence: `report/extracted_sources/REPORT.md.txt`; `report/extracted_sources/EmoFlow_Presentation.pptx.txt`.
- Safer rewrite: "The reported per-class analysis identifies fear as unresolved, with no successful test predictions in that run."

### "The reference papers fully support all related-work claims."

- Risk: Papers are extracted, but final citation metadata and exact quote/page references still need verification.
- Evidence: `report/source_inventory.md`; `report/related_work_plan.md`.
- Safer rewrite: "Extracted reference papers support these planning notes; final citation metadata should be verified before drafting."

## Safe Claims

### EmoFlow is a classifier for conversational emotion recognition.

- Evidence: `report/extracted_sources/REPORT.md.txt`; `report/extracted_sources/README.md.txt`; `report/extracted_sources/model.py.txt`.
- Suggested wording: "EmoFlow is a classifier for conversational emotion recognition."

### The implemented model separates encoder, temporal memory, and BayesianHead modules.

- Evidence: `report/extracted_sources/model.py.txt`; `report/extracted_sources/encoder.py.txt`; `report/extracted_sources/memory.py.txt`; `report/extracted_sources/bayes.py.txt`.
- Suggested wording: "The implementation separates utterance encoding, temporal memory, and prior-likelihood fusion."

### TemporalMemory is causal and uses exponential decay over prior turns.

- Evidence: `report/extracted_sources/memory.py.txt`.
- Suggested wording: "TemporalMemory uses causal exponential-decay weighting over current and previous appraisals."

### The appraisal targets are 8-dimensional in implementation.

- Evidence: `report/extracted_sources/appraisal_targets.py.txt`; `report/extracted_sources/encoder.py.txt`.
- Suggested wording: "The implemented appraisal representation has 8 dimensions."

### LoRA is implemented in the encoder.

- Evidence: `report/extracted_sources/encoder.py.txt`; `report/extracted_sources/Paper_LoRA.pdf.txt`.
- Suggested wording: "The encoder uses LoRA adapters for parameter-efficient adaptation."

### Weighted F1 and ETA are implemented metrics.

- Evidence: `report/extracted_sources/evaluation.py.txt`; `report/extracted_sources/train.py.txt`.
- Suggested wording: "We report weighted F1 and Emotion Transition Accuracy."

### The repository does not include raw datasets and checkpoints.

- Evidence: `report/extracted_sources/README.md.txt`; `report/source_inventory.md`.
- Suggested wording: "Raw datasets and checkpoints are not included in the repository snapshot."

## Suggested Safer Rewrites Summary

| Risky wording | Safer wording |
| --- | --- |
| "fully Bayesian" | "Bayesian-inspired" or "Bayesian-style prior-likelihood fusion" |
| "outperforms published ERC baselines" | "outperforms the implemented baselines under our in-repository matched setup" |
| "state-of-the-art" | Avoid; use "competitive in the documented project setting" only with caveats |
| "ground-truth appraisal annotations" | "weakly supervised appraisal targets derived from emotion prototypes" |
| "response generation was completed" | "response generation was proposed but deferred" |
| "fully reproducible from the repository" | "documented in project reports, but checkpoints and processed data are not fully included" |
| "architecture alone achieves 0.62" | "0.62 is reported with selective DailyDialog rare-class augmentation" |
| "LSTM models are worse than EmoFlow" | "the implemented BiLSTM-over-appraisals baseline underperformed EmoFlow in the matched ablation" |
