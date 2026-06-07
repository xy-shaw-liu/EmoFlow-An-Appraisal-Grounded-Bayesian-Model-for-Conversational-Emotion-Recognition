# Missing or Uncertain Information

This audit is based on `report/outline.md`, `report/related_work_plan.md`, `report/source_summaries.md`, `report/source_inventory.md`, and extracted sources under `report/extracted_sources/`.

## Citation and Bibliography Gaps

- Exact BibTeX metadata is still needed for Scherer 2001, Scherer 1997/ISEAR, MELD, DialogueRNN, DialogueGCN, COSMIC, MMGCN, LoRA, MKE-IGN, Chain-of-Emotion, EAI, and the LLM emotion interpretability paper.
- Page numbers for Scherer Table 5.5 and any ISEAR-derived appraisal targets need verification from the original PDF or reliable citation metadata.
- The ACL template bibliography files in the extracted sources are mostly placeholders or sample entries and should not be treated as final project bibliography.
- Published baseline numbers in REPORT.md need citation verification against the original papers before final ACL prose.

## Dataset and Split Details

- Exact train/dev/test split statistics differ across project notes and reference papers and should be reconciled before final writing.
- MELD counts are reported in project files and MKE-IGN, but the final report should choose one verified source and cite it.
- Processed data under `data_processed/` is not included in the extracted inventory.
- Raw datasets under `datasets/` are skipped/excluded; the repository does not ship raw MELD, EmoryNLP, or DailyDialog data.
- It is uncertain whether every reported dataset statistic can be regenerated without downloading external data.

## Experimental Reproducibility Limits

- Raw checkpoint directories such as `ckpt/` and `ckpt_lambda/` are skipped/gitignored and not included in the repository snapshot.
- Final result claims are supported by REPORT.md, README.md, and slides, but not independently verifiable from included checkpoint metrics.
- `make_table.py` expects `metrics.json` files under checkpoint directories, but those files are not included.
- Training outputs such as `args.json`, `train_log.jsonl`, `best_model.pt`, `metrics.json`, and `predictions.csv` are described but absent from the extracted source set.
- It is uncertain whether all reported results can be regenerated exactly without access to the original datasets, checkpoints, run logs, hardware, and model permissions.

## Model and Hyperparameter Details

- Exact trainable parameter counts should be verified from the final run, not only reported approximately as ~20M.
- Exact total parameter counts depend on the backbone and LoRA configuration used in the final run.
- The README and slides report a final configuration with LLaMA-3-8B, LoRA rank 8, batch size 2, learning rate 5e-4, 3 epochs, alpha 0.1, seed 42, and threshold 0.2, but raw run args from the final checkpoint are not included.
- Hardware details are partly reported in README/slides, but exact final training hardware, runtime, CUDA setup, and memory usage are not independently recorded in included run artifacts.
- The impact of optional `bce_pos_weight`, label smoothing, or appraisal pretraining on final runs should not be inferred unless tied to a documented command or metrics file.

## Method-Specification Gaps and Contradictions

- `dataloader.py` contains stale comments/docstrings mentioning 5-dimensional appraisal targets, while `appraisal_targets.py`, `encoder.py`, REPORT.md, README.md, and slides indicate the implemented representation is 8-dimensional.
- The claim that log-additive MLP fusion is mathematically equivalent to Dirichlet-Categorical updating should be softened unless a formal derivation is included.
- EmoFlow should be described as Bayesian-inspired or Bayesian-style, not fully Bayesian.
- Appraisal targets are emotion-prototype weak supervision, not utterance-level ground-truth psychological annotations.
- Surprise appraisal targets are derived theoretically in code because ISEAR did not include surprise.

## Experimental Completeness Gaps

- Final post-fix EmoryNLP results are missing.
- IEMOCAP was not attempted.
- Response generation was proposed but deferred.
- Human Coherence Score evaluation was proposed but not performed.
- MELD audio/video modalities are unused; experiments are text-only.
- The fear class remains unresolved in the reported analysis.
- Appraisal-only DailyDialog pretraining is implemented but reportedly not used in the final pipeline.

## Source-Extraction Notes

- `report/source_inventory.md` now shows successful extraction of core documents, slides, PDFs, and reference papers.
- `report/source_summaries.md` contains stale statements saying some PDFs/DOCX/PPTX files were empty or unreadable; these should be corrected in a later cleanup or ignored in favor of `source_inventory.md`.
- The current related-work plan has already been corrected to use the successfully extracted reference paper text.

## Suggested Honest Phrasing for Incomplete Evidence

- "Under our in-repository setup..."
- "The project report documents..."
- "Preliminary results suggest..."
- "We observe in the matched ablation setting..."
- "The included repository does not contain the raw checkpoints needed to independently verify this number."
- "We do not claim state-of-the-art performance over published MELD systems."
- "Response generation remains future work."
- "The classifier uses a Bayesian-inspired prior-likelihood factorization rather than full Bayesian inference."
