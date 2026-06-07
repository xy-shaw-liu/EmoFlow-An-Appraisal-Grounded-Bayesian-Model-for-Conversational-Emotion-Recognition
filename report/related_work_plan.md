# Related Work Plan

This file organizes related work into theme clusters for the EmoFlow ACL-style report. It is a planning artifact, not final prose.

`report/source_inventory.md` shows that all six `Paper_*.pdf` reference papers were successfully extracted into text with substantial length. Use the extracted paper text as readable planning evidence, while still verifying final BibTeX metadata before drafting citations.

## Reference Paper Extraction Status

- `Paper_Appraisal_Processes_in_Emotion_Theory_Methods_Rese..._----_(5_Appraisal_Considered_as_a_Process_of_Multilevel_Sequential_Checking).pdf.txt`: success, Scherer chapter on the Component Process Model and sequential appraisal checks.
- `Paper_chain-of-emotion_for_affective_LM.pdf.txt`: success, appraisal-based Chain-of-Emotion architecture for affective LLM game agents. Citation metadata to verify.
- `Paper_Emotional_Decision_Making_of_LLMs.pdf.txt`: success, EAI framework for emotional effects on LLM decision-making in strategic games and ethical dilemmas. Citation metadata to verify.
- `Paper_Interpretability_of_Emotion_Inference_in_LLM.pdf.txt`: success, mechanistic interpretability of emotion inference in LLMs. Citation metadata to verify.
- `Paper_LoRA.pdf.txt`: success, LoRA low-rank adaptation paper. Citation metadata to verify.
- `Paper_Multi_GraphNetwork_for_Multi_Emotion_Recognition.pdf.txt`: success, Multiple Knowledge-Enhanced Interactive Graph Network for multimodal ERC. Citation metadata to verify.

## 1. Conversational Emotion Recognition

### Relevant Sources

- `REPORT.md.txt`
- `README.md.txt`
- `EmoFlow_Presentation.pptx.txt`
- `evaluation.py.txt`
- `baselines.py.txt`
- `Paper_Multi_GraphNetwork_for_Multi_Emotion_Recognition.pdf.txt`

### Planning Notes from Reference Papers

- The MKE-IGN paper defines multimodal ERC as identifying emotions in conversational videos and states that current work often models context-sensitive and speaker-sensitive dependencies plus multimodal fusion.
- The MKE-IGN paper frames ERC as challenging because conversation is dynamic and spontaneous, with individuals expressing varied emotions across turns.
- Its related-work section separates context-sensitive models, speaker-sensitive models, and knowledge-sensitive models, which can help structure the ERC background paragraph.
- It reports MELD as a multiparty Friends-based dataset with 1,433 conversations, 13,708 utterances, and 304 speakers; verify whether to use these numbers or the project-specific counts from REPORT/README.

### EmoFlow Positioning Notes

- EmoFlow is text-only, while MKE-IGN is multimodal; use this to motivate EmoFlow's limitation rather than overstate comparison.
- EmoFlow models context through appraisal memory rather than graph-based multimodal/common-sense fusion.
- Avoid claiming superiority over MKE-IGN unless exact experimental settings and metrics are verified.

## 2. Appraisal Theory and Cognitive Emotion Modeling

### Relevant Sources

- `REPORT.md.txt`
- `README.md.txt`
- `appraisal_targets.py.txt`
- `emoflow_algorithm_explained.md.txt`
- `EmoFlow_Presentation.pptx.txt`
- `Paper_Appraisal_Processes_in_Emotion_Theory_Methods_Rese...pdf.txt`
- `Paper_chain-of-emotion_for_affective_LM.pdf.txt`
- `Paper_Interpretability_of_Emotion_Inference_in_LLM.pdf.txt`

### Planning Notes from Reference Papers

- Scherer's chapter presents the Component Process Model and sequential check theory, where emotion differentiation results from sequences of stimulus evaluation/appraisal checks.
- Scherer describes emotion as synchronized changes across organismic subsystems in response to evaluation of an internal or external event as relevant to the organism.
- Scherer's appraisal checks are grouped into relevance, implications, coping potential, and normative significance; this supports EmoFlow's use of interpretable appraisal dimensions.
- The Chain-of-Emotion paper argues that appraisal is a central feature for simulating affect because emotions depend on subjective evaluations of events, including goal relevance, certainty, coping potential, and agency.
- The LLM interpretability paper uses cognitive appraisal theory to test whether LLM emotion representations are psychologically plausible.

### EmoFlow Positioning Notes

- EmoFlow operationalizes appraisal theory as an 8-dimensional weak-supervision bottleneck derived in `appraisal_targets.py`.
- The final paper should distinguish Scherer's broader appraisal objectives from the project's specific 8 ISEAR-derived dimensions.
- The surprise profile is project-derived, not directly reported as an ISEAR empirical profile.

## 3. Bayesian or Probabilistic Emotion Modeling

### Relevant Sources

- `REPORT.md.txt`
- `bayes.py.txt`
- `model.py.txt`
- `emoflow_algorithm_explained.md.txt`
- `EmoFlow_Presentation.pptx.txt`
- `Paper_Emotional_Decision_Making_of_LLMs.pdf.txt`
- `Paper_chain-of-emotion_for_affective_LM.pdf.txt`

### Planning Notes from Reference Papers

- The EAI paper motivates emotion modeling as important for predicting LLM behavior in decision-making settings, especially because emotions can shift ethical and strategic choices.
- EAI studies emotional effects in strategic games and ethical benchmarks; use this as broader support that emotion modeling can matter beyond label prediction.
- Chain-of-Emotion discusses affective agents as computational operationalizations of emotion-theoretical models with appraisal functionality for emotion generation.
- Chain-of-Emotion notes that building fully developed psychology-based emotion simulations remains difficult, supporting EmoFlow's conservative approximation stance.

### EmoFlow Positioning Notes

- EmoFlow is Bayesian-inspired, not fully Bayesian: `bayes.py` implements two MLPs whose logits are added as prior and likelihood terms.
- The proposed explicit Dirichlet-Categorical update was not implemented; describe it as planned/proposed or future work.
- Soften any claim of exact probabilistic equivalence unless the paper includes a derivation.

## 4. Emotion Flow / Temporal Emotion Dynamics

### Relevant Sources

- `REPORT.md.txt`
- `README.md.txt`
- `memory.py.txt`
- `model.py.txt`
- `evaluation.py.txt`
- `EmoFlow_Presentation.pptx.txt`
- `Paper_chain-of-emotion_for_affective_LM.pdf.txt`
- `Paper_Multi_GraphNetwork_for_Multi_Emotion_Recognition.pdf.txt`

### Planning Notes from Reference Papers

- Chain-of-Emotion focuses on affective agents whose emotional responses unfold through language-mediated appraisal and agent interaction, making it relevant to temporal affect simulation.
- Chain-of-Emotion discusses language-model agents using memory, reflection, plans, and actions in simulated worlds; this is useful background for the proposed-but-deferred response generation direction.
- MKE-IGN notes that some multimodal ERC methods rely on future utterances to predict the current emotion, which it calls impractical for real-world settings.
- MKE-IGN uses a directed graph structure to avoid future-utterance leakage; this is a useful contrast with EmoFlow's causal TemporalMemory.

### EmoFlow Positioning Notes

- EmoFlow's TemporalMemory is causal and exponential-decay based, using only current and previous turns.
- The no-decay and learned-lambda ablations can be framed as tests of temporal weighting assumptions.
- Do not generalize from EmoFlow's LSTM result to all recurrent ERC models; keep it tied to the implemented BiLSTM-over-appraisals baseline.

## 5. Dialogue Context Modeling

### Relevant Sources

- `REPORT.md.txt`
- `README.md.txt`
- `dataloader.py.txt`
- `memory.py.txt`
- `baselines.py.txt`
- `evaluation.py.txt`
- `EmoFlow_Presentation.pptx.txt`
- `Paper_Multi_GraphNetwork_for_Multi_Emotion_Recognition.pdf.txt`
- `Paper_Interpretability_of_Emotion_Inference_in_LLM.pdf.txt`

### Planning Notes from Reference Papers

- MKE-IGN states that textual ERC commonly models context-sensitive and speaker-sensitive dependencies.
- Its related-work section notes RNN-based context models, memory networks, speaker-specific models, and graph-based models as major context-modeling families.
- MKE-IGN integrates textual and visual commonsense knowledge into edge representations to model relations between utterances and commonsense knowledge.
- The LLM interpretability paper gives examples where different appraisal interpretations of a situation can distinguish emotions such as guilt and sadness; use this to motivate context-sensitive appraisal rather than bag-of-words emotion labeling.

### EmoFlow Positioning Notes

- EmoFlow uses dialogue-level batching and masks to preserve turn order and context.
- EmoFlow's context state is an aggregated history of appraisal vectors, not raw utterance embeddings or multimodal features.
- Dialogue-level oversampling in `train.py` preserves sequence context; mention as an implementation detail in experiments rather than related work.

## 6. Neural Baselines or LLM-Based Emotion Recognition

### Relevant Sources

- `REPORT.md.txt`
- `README.md.txt`
- `encoder.py.txt`
- `train.py.txt`
- `EmoFlow_Presentation.pptx.txt`
- `Paper_LoRA.pdf.txt`
- `Paper_Emotional_Decision_Making_of_LLMs.pdf.txt`
- `Paper_Interpretability_of_Emotion_Inference_in_LLM.pdf.txt`
- `Paper_chain-of-emotion_for_affective_LM.pdf.txt`

### Planning Notes from Reference Papers

- LoRA freezes pretrained model weights and injects trainable low-rank matrices into Transformer layers, reducing trainable parameters for downstream adaptation.
- LoRA reports large reductions in trainable parameters and GPU memory relative to full fine-tuning, with no additional inference latency after merging low-rank updates.
- The LLM interpretability paper states that LLMs can predict human emotions from text, but internal mechanisms of emotion processing are underexplored.
- That paper probes hidden representations, finds functional localization of emotion-related activations, and uses appraisal concept interventions to steer output emotion.
- EAI argues that NLP-only evaluation is insufficient for agentic LLMs because emotions can affect strategic and ethical behavior.
- Chain-of-Emotion provides evidence that LLMs can be organized with appraisal-based prompting/architecture for affective agents, but its domain is game-agent emotion simulation rather than ERC classification.

### EmoFlow Positioning Notes

- EmoFlow uses LoRA/QLoRA for parameter-efficient adaptation of a frozen LLaMA-3-8B encoder.
- EmoFlow differs from black-box prompting work by training an explicit appraisal bottleneck and downstream temporal/Bayesian classifier.
- Keep LLM emotion and affective-agent papers as motivation and adjacent work, not direct MELD baselines.

## 7. Multi-Label Emotion Recognition

### Relevant Sources

- `REPORT.md.txt`
- `README.md.txt`
- `train.py.txt`
- `EmoFlow_Presentation.pptx.txt`
- `Paper_Multi_GraphNetwork_for_Multi_Emotion_Recognition.pdf.txt`
- `Paper_Emotional_Decision_Making_of_LLMs.pdf.txt`

### Planning Notes from Reference Papers

- MKE-IGN's task definition is utterance-level emotion-label prediction for each utterance in a conversation; it does not establish EmoFlow's multilabel reformulation by itself.
- MKE-IGN reports seven classes for MELD and six for IEMOCAP, reinforcing that standard ERC benchmarks are usually single-label classification.
- EAI uses controlled emotion states/prompts to study shifts in decisions, which is adjacent evidence that emotional categories can be modeled as intervention conditions, not direct evidence for multilabel ERC.
- No extracted reference paper currently provides direct support that MELD should be treated as a true multilabel dataset.

### EmoFlow Positioning Notes

- Treat EmoFlow's six-way BCE formulation as a project-specific reformulation of a single-label task.
- State neutral-as-all-zero and threshold decoding as an implementation choice supported by `train.py`, REPORT, README, and slides.
- Avoid implying that MELD annotations are multi-label.

## Cross-Cutting Citation Gaps

- Verify final BibTeX metadata for Scherer 2001, Chain-of-Emotion, EAI, Mechanistic Interpretability of Emotion Inference in LLMs, LoRA, MKE-IGN, MELD, DialogueRNN, DialogueGCN, COSMIC, and MMGCN.
- Do not use template bibliography entries from `custom.bib` as EmoFlow related work unless they are actually relevant.
- Published baseline numbers must remain caveated because EmoFlow's headline run uses selective DailyDialog rare-class augmentation, while many published MELD baselines train on MELD-only settings.
- For every final related-work claim, cite either extracted paper text, project files, or verified bibliography metadata.
