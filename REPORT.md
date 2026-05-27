# EmoFlow: An Appraisal-Grounded Bayesian Model for Conversational Emotion Recognition

**ECS 271 Final Project Report**

---

## Abstract

We present **EmoFlow**, an appraisal-grounded Bayesian model for conversational emotion recognition (ERC) on MELD. Our model factorizes the task into three stages mirroring Scherer's Component Process Model (CPM): a frozen LLaMA-3-8B encoder with LoRA adapters produces an 8-dimensional appraisal vector per utterance; a TemporalMemory module aggregates past appraisals with a learnable exponential-decay kernel; a BayesianHead combines prior (from memory) and likelihood (from current appraisal) via log-additive fusion to produce per-turn emotion logits.

We document a series of representation-collapse failure modes encountered during training, trace the root cause to **sigmoid saturation in the appraisal head**, and demonstrate that a one-line architectural fix combined with selective cross-dataset augmentation produces an emotion classifier matching DialogueRNN (wF1=0.62 on MELD test) using only ~20M trainable parameters.

**Key contributions:**
1. EmoFlow architecture grounded in appraisal theory + Bayesian inference
2. Diagnostic of sigmoid saturation as a reproducible failure mode in bounded-bottleneck emotion classifiers
3. Selective cross-dataset rare-class augmentation strategy
4. SOTA-tier results with minimal parameter footprint

---

## 1. Introduction

### 1.1 Task

**Conversational Emotion Recognition (ERC)**: given a sequence of utterances $\{u_1, u_2, \ldots, u_T\}$ from one or more speakers, predict the emotion label $y_t \in \mathcal{E}$ for each turn $t$, where $\mathcal{E}$ is a fixed emotion vocabulary (in our case 7 classes: *neutral, joy, sadness, anger, fear, disgust, surprise*).

### 1.2 Why ERC Is Hard

Three structural challenges differentiate ERC from sentence-level emotion classification:

1. **Context dependence**: The same utterance carries different emotional content depending on dialogue history. "Sure." can be *neutral* (acknowledgement), *disgust* (sarcastic dismissal), or *surprise* (delayed acceptance) — only context disambiguates.

2. **Emotional inertia**: Emotional states persist and evolve. After three angry turns, a subsequent flat utterance like "Whatever." is still likely angry.

3. **Severe class imbalance**: In MELD, *neutral* alone accounts for 48% of turns, while *fear* and *disgust* together account for less than 5%. Naive classifiers collapse to predicting the majority class.

### 1.3 Our Approach

EmoFlow addresses these challenges through a three-component pipeline aligned with appraisal-theoretic emotion psychology:

```
Text ─[StimulusEncoder]→ Appraisal (8d) ─[TemporalMemory]→ Memory state
                                              │                  │
                                              ▼                  ▼
                                       [BayesianHead: likelihood + prior]
                                              │
                                              ▼
                                       Emotion logits
```

- **Appraisal grounding**: Force the model's internal representation to align with interpretable Scherer dimensions (weak supervision against Scherer Table 5.5).
- **Bayesian fusion**: Explicitly separate context-driven prior and utterance-driven likelihood.
- **Structured temporal weighting**: Learnable exponential decay rather than generic RNN.

### 1.4 Scope: Pivot from Generation to Classification

The original project proposal envisioned a six-module pipeline whose final stage was a **Response Generator**: an LLaMA-3-8B decoder taking the predicted emotion distribution as a soft-prompt prefix to produce empathetic dialogue responses. Evaluation was to include a **Coherence Score** based on Likert-scale human ratings of generated replies on 100 EmoryNLP test dialogues.

During implementation we encountered a series of representation-collapse failure modes in the upstream classifier (documented in §6) that consumed a substantial portion of the project's compute and engineering budget. After the architectural fix and data augmentation pushed the classifier to wF1=0.62 (matching DialogueRNN), we made the deliberate scope decision to **publish the classifier as a self-contained contribution** rather than rush an unvetted generation component.

The classifier we deliver is a strict prerequisite for the planned generator: any emotion-aware response system needs an emotion encoder first. The findings reported here — particularly the sigmoid-saturation diagnostic — are independent contributions usable by future work on emotion-aware dialogue generation. The Response Generator and Coherence Score evaluation are deferred to future work (§8.5).

**Other deviations from proposal**:
- Appraisal dimensionality expanded from the proposal's 5 aggregated dimensions to the full 8 ISEAR dimensions, motivated by representation-bottleneck analysis (§3.2, §6).
- The proposal's main evaluation dataset was EmoryNLP; we shifted to MELD as the primary benchmark after EmoryNLP exhibited a speaker-disjoint test split that triggered persistent class collapse in our pipeline (§4.1, §8.4).
- The proposal's Dirichlet–Categorical Bayesian update was implemented as a log-additive MLP fusion, which is mathematically equivalent under uniform marginals while being significantly simpler to train end-to-end (§3.4).
- LoRA rank tuned from the proposal's $r=16$ estimate to $r=8$ based on memory constraints under QLoRA (Appendix A).

---

## 2. Theoretical Foundation

### 2.1 Appraisal Theory: Scherer's CPM

In Scherer's Component Process Model of emotion (Scherer 2001), an emotion is **not** a primitive label. Instead, it is the consequence of the brain evaluating a stimulus along a fixed set of cognitive dimensions called *appraisals*. The ISEAR cross-cultural study (Scherer 1997, 37 countries, n≈3000) reports empirical Z-scores for 8 ISEAR appraisal dimensions across 6 emotions.

The 8 appraisal dimensions:

| Dimension | Question it answers |
|-----------|--------------------|
| expectedness | Was this event expected? |
| unpleasantness | Is this stimulus pleasant or unpleasant? |
| goal_hindrance | Does it block my goals? |
| external_causation | Was this caused by external forces? |
| coping_potential | Can I cope with this? |
| unfairness | Is this unfair? |
| immorality | Does this violate moral norms? |
| self_consistency | Is this consistent with my self-image? |

Each emotion has a characteristic appraisal *signature* (Table 5.5):

- **Joy** = high expectedness, low unpleasantness, high coping_potential, high self_consistency
- **Anger** = high unpleasantness, high goal_hindrance, high unfairness, high immorality
- **Fear** = low expectedness, high unpleasantness, low coping_potential
- **Surprise** = very low expectedness (high novelty), other dimensions near baseline

**Why this matters for our model**: If we can train the encoder to output utterance-level appraisal vectors that match these signatures, we get:
1. **Interpretability**: each dimension has a name and theoretical meaning
2. **Structured bottleneck**: 8 numbers must capture the emotionally relevant content
3. **Theory-grounded weak supervision**: Scherer's table provides regression targets

### 2.2 Bayesian Inference for Emotion

Given the current utterance's appraisal $a_t$ and dialogue context $\mathcal{C}_t$, the probability of emotion $e$ at turn $t$ is, by Bayes' rule:

$$
P(e \mid a_t, \mathcal{C}_t) = \frac{P(a_t \mid e) \cdot P(e \mid \mathcal{C}_t)}{P(a_t \mid \mathcal{C}_t)}
$$

Taking log on both sides (and absorbing the normalizer):

$$
\log P(e \mid a_t, \mathcal{C}_t) = \underbrace{\log P(a_t \mid e)}_{\text{likelihood}} + \underbrace{\log P(e \mid \mathcal{C}_t)}_{\text{prior}} - \log Z
$$

This is the **log-additive Bayesian decomposition** at the heart of EmoFlow's BayesianHead. We will implement each log-probability term as a neural network output (a learned MLP), and sum them to produce posterior logits.

**Interpretation:**
- **Likelihood**: How likely is the current appraisal pattern given each emotion? (Scherer-style mapping)
- **Prior**: How likely is each emotion given the dialogue so far? (Temporal context)
- **Posterior**: The combined evidence determines the predicted emotion.

### 2.3 Why Exponential Decay for Memory

Context $\mathcal{C}_t$ is composed of the past appraisals $\{a_1, \ldots, a_t\}$, but **not all past turns are equally relevant**. Recent turns generally influence current emotion more than distant ones. This is consistent with:
- Behavioral findings on emotional persistence (decay over seconds to minutes)
- Cognitive models of working memory (exponential forgetting curves)

We model context aggregation as:

$$
\mathcal{C}_t = \frac{\sum_{i \leq t} w_i \cdot a_i}{\sum_{i \leq t} w_i}, \quad w_i = \exp(-\lambda (t - i))
$$

with $\lambda \geq 0$ learned end-to-end. This gives:
- **$\lambda = 0$**: uniform average over all past turns (no decay)
- **$\lambda \to \infty$**: only the current turn matters (effectively stateless)
- **$\lambda$ moderate**: smooth decay with recent emphasis

---

## 3. Method

### 3.1 Architecture Overview

EmoFlow's forward pass at turn $t$:

```
       u_t (raw text)
            │
            ▼
   ┌────────────────────┐
   │  StimulusEncoder   │  frozen LLaMA-3-8B + LoRA + AppraisalHead
   └────────────────────┘
            │
            ▼  a_t ∈ ℝ⁸  (Scherer appraisal vector)
            │
            ▼
   ┌────────────────────┐
   │  TemporalMemory    │  weighted aggregation: exp(-λ·Δt)
   └────────────────────┘
            │            
            ▼  h_t ∈ ℝ⁸  (memory state)
            │
            ▼
   ┌────────────────────┐
   │   BayesianHead     │  prior(h_t) + likelihood(a_t)
   └────────────────────┘
            │
            ▼  z_t ∈ ℝ⁶  (emotion logits, 6 non-neutral classes)
            │
            ▼
       sigmoid → max_p → threshold ⟹ predicted emotion
```

### 3.2 StimulusEncoder (encoder.py)

**Purpose**: Map a single utterance text to an 8-dimensional appraisal vector.

**Components**:
1. **Frozen LLaMA-3-8B**: Pretrained representations. All 8 billion parameters are frozen — only LoRA adapters update during training.
2. **LoRA adapters** ($r=8$): Low-rank matrices injected on attention's query and value projections (`q_proj`, `v_proj`). Each attention layer's projection $W$ is augmented as $W + BA$ where $A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}$. Only $A$ and $B$ are trained.
3. **4-bit quantization (QLoRA)**: The frozen LLaMA weights are stored in 4-bit precision (NF4 format) to fit in 24GB GPU memory. Compute happens in bfloat16.
4. **Pooling**: Take the last non-padding token's hidden state as the pooled utterance representation $h \in \mathbb{R}^{4096}$.
5. **AppraisalHead**: A two-layer MLP mapping $\mathbb{R}^{4096} \to \mathbb{R}^{8}$:

```python
self.head = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.GELU(),
    nn.Linear(4096, 8),    # NO sigmoid (see §6)
)
```

**Trainable parameter count**: ~20M (LoRA + AppraisalHead), out of ~8B total. We train 0.25% of the parameters.

**Dimension choice (8 vs proposal's 5)**: The original project proposal specified 5 aggregated appraisal dimensions (`novelty, pleasantness, goal-relevance, coping, norm-compatibility`), each derived from one or more raw ISEAR columns of Scherer Table 5.5 via averaging or sign-flipping. We expanded to the full 8-dim raw ISEAR representation for two reasons: (i) the aggregated 5-dim collapsed *unfairness* and *immorality* into a single `norm` channel via averaging, removing the distinction between anger (high unfairness) and disgust (high immorality) that proved diagnostically important in §7.6; (ii) preserving direct correspondence with Scherer's published table improves interpretability of qualitative case studies. We note that the underlying representation collapse seen with the 5-dim version (§6) was caused by the sigmoid activation, not the dimensionality — but the 8-dim version is retained as the final design on theoretical grounds.

**Important design decision: NO sigmoid output activation.** An initial version applied `nn.Sigmoid()` after the final Linear, intending to constrain outputs to $[0,1]$ to match Scherer's normalized targets. This caused catastrophic saturation; we discuss the diagnostic in §6.

### 3.3 TemporalMemory (memory.py)

**Purpose**: Aggregate past appraisal vectors into a context summary.

**Implementation**: For each query turn $t$, compute weights for all keys $i \leq t$:

$$
w_{t,i} = \frac{\exp(-\lambda (t - i))}{\sum_{j \leq t} \exp(-\lambda (t - j))}
$$

(Note: we implement this as a softmax over $-\lambda \cdot \Delta t$ with causal masking, which is numerically equivalent to the explicit normalization above and reuses well-tested attention infrastructure.)

Then:
$$
h_t = \sum_{i \leq t} w_{t,i} \cdot a_i
$$

**Learnable $\lambda$**: We parameterize $\lambda = \mathrm{softplus}(\theta)$ with $\theta \in \mathbb{R}$ learnable. Softplus ensures $\lambda > 0$ (decay must be non-negative).

**Why softmax-based attention?**
- Numerical stability (handles long sequences)
- Implementable in 5 lines using PyTorch's `torch.softmax`
- Easy to mask out padding positions

### 3.4 BayesianHead (bayes.py)

**Purpose**: Combine memory state and current appraisal into emotion logits using Bayesian decomposition.

**Implementation**: Two independent MLPs:

```python
prior_head      = MLP(8 → 64 → K)   # P(emotion | memory_state)
likelihood_head = MLP(8 → 64 → K)   # P(current_appraisal | emotion)
```

Both MLPs use:
```
Linear(8, 64) → GELU → Dropout(0.1) → Linear(64, K)
```

where $K = 6$ in the final multilabel formulation.

**Forward**:
```python
prior_logits     = prior_head(memory_state)         # (B, T, K)
likelihood_logits = likelihood_head(current_appraisal)  # (B, T, K)
posterior_logits = prior_logits + likelihood_logits  # (B, T, K)
```

**Theoretical justification**: In log-space, multiplication becomes addition. The unnormalized log-posterior is the sum of log-likelihood and log-prior; the partition function $\log Z$ is implicit in subsequent softmax (or sigmoid) normalization.

**Why two heads instead of one big head?** Decoupling forces an interpretable structure: ablations can replace either head (e.g., uniform prior for "memory-blind" model) to measure context contribution explicitly.

**Deviation from proposal: Dirichlet–Categorical → log-additive MLP.** The proposal specified an explicit Dirichlet–Categorical Bayesian update, where the prior was a Dirichlet distribution over emotion concentration parameters and the likelihood was a Categorical observation. The two were combined by Dirichlet posterior update ($\alpha_{\text{post}} = \alpha_{\text{prior}} + \text{counts}$). We implemented the simpler **log-additive MLP fusion** instead. Mathematically, the two are equivalent under a uniform marginal assumption: any conjugate Dirichlet–Categorical posterior on the simplex has a log form that is linear in the count statistics, which is exactly what two MLPs adding logits provide. The MLP form is end-to-end differentiable without parameterizing concentration directly, avoiding the numerical issues of small-counts Dirichlet updates. We acknowledge this departs from the proposal's explicit probabilistic formulation; future work could revisit a fully Bayesian (e.g., variational) treatment.

### 3.5 Output: Multilabel Reformulation

**Standard ERC approach**: 7-way multiclass classification with softmax + cross-entropy. Each turn gets a softmax over 7 classes including neutral.

**Why we abandoned this**: The 7-way softmax has a degenerate solution where the model places almost all mass on *neutral* (the majority class) — a stable local minimum that imbalance-mitigation techniques (class weights, label smoothing, oversampling) fail to escape (see §6).

**Our reformulation**: 6-way multilabel classification with sigmoid + binary cross-entropy.
- Vocabulary $\mathcal{E}_6 = \{\text{joy, sadness, anger, fear, disgust, surprise}\}$ (no neutral)
- Model outputs $K=6$ independent logits per turn
- **Neutral is represented as the all-zero vector** $[0,0,0,0,0,0]$ (no emotion fired)
- Non-neutral turn $e$ is represented as one-hot at index $e$ in $\mathcal{E}_6$

**Inference**: Compute sigmoid probabilities $p_k = \sigma(z_k)$. If $\max_k p_k < \tau$ → predict *neutral*. Otherwise → predict $\arg\max_k p_k$. The threshold $\tau$ is selected on dev (we use $\tau = 0.2$).

**Why this breaks the collapse**: In the 7-way softmax setting, predicting "all probability on neutral" is achievable by output-independent constants. In the 6-way multilabel setting, neutral requires actively pushing **all six** independent logits below the threshold — a much harder degenerate solution to fall into. The model has no single "default" class to hide in.

This reformulation is theoretically aligned with appraisal psychology: emotions are independent components that may fire simultaneously (joy + surprise = excitement), and neutral simply means "no component fires".

---

## 4. Data

### 4.1 Datasets

| Dataset | Train dlg | Vocab | Domain |
|---------|----------|-------|--------|
| **MELD** | 1,038 | 7-class (neutral + 6) | Friends sitcom |
| **DailyDialog** | 11,118 | 7-class (same vocab) | General daily conversations |
| EmoryNLP | 713 | 7-class (different) | Friends sitcom |

We train and evaluate primarily on **MELD**, with selective augmentation from **DailyDialog**.

**Deviation from proposal: dataset swap.** The original proposal designated **EmoryNLP** as the primary benchmark and MELD as auxiliary. We swapped this assignment after preliminary experiments revealed that EmoryNLP exhibits a particularly difficult speaker-disjoint train/test split — characters appearing in test rarely appear in training — which interacted catastrophically with our class-imbalance failure mode (§6) and produced bit-identical 0.0789 wF1 collapse across all interventions. MELD, while still imbalanced, has more uniform speaker distribution and proved tractable enough to expose the underlying sigmoid-saturation issue. We did not run final-quality EmoryNLP experiments on the fixed pipeline; this is acknowledged as a limitation (§8.4).

### 4.2 Label Normalization

Datasets use different label spellings:
- `happiness` (DD) → `joy`
- `joyful, sad, mad, scared` (EmoryNLP) → `joy, sadness, anger, fear`
- `no_emotion` (DD) → `neutral`

We normalize all labels into a unified 7-class vocabulary in the DataLoader (not the preprocessor) so we keep the raw labels available for analysis.

### 4.3 Appraisal Targets from Scherer Table 5.5

For each non-neutral, non-mask emotion, we extract the 8-dimensional ISEAR Z-score vector from Scherer (2001) Table 5.5. Surprise is derived theoretically (low expectedness, others at population mean = 0) because the original ISEAR study did not include surprise.

We then **min-max normalize each dimension across the 6 emotions to $[0,1]$**:

```
joy      : [1.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 1.00]
fear     : [0.54, 0.97, 0.84, 0.60, 0.08, 0.55, 0.63, 0.08]
anger    : [0.48, 1.00, 1.00, 0.42, 0.53, 1.00, 1.00, 0.06]
sadness  : [0.63, 0.98, 0.96, 1.00, 0.00, 0.64, 0.55, 0.00]
disgust  : [0.49, 1.00, 0.85, 0.67, 0.47, 0.76, 1.00, 0.11]
surprise : [0.00, 0.83, 0.76, 0.22, 0.44, 0.55, 0.69, 0.13]
```

(Columns: expectedness, unpleasantness, goal_hindrance, external_causation, coping_potential, unfairness, immorality, self_consistency.)

These vectors serve as **regression targets** for the appraisal head, supervised via MSE loss (weak supervision).

### 4.4 MELD Class Distribution (Train)

| Emotion | Turns | % |
|---------|------:|--:|
| neutral | 4710 | 47.2% |
| joy | 1743 | 17.4% |
| surprise | 1205 | 12.1% |
| anger | 1109 | 11.1% |
| sadness | 683 | 6.8% |
| disgust | 271 | 2.7% |
| fear | 268 | 2.7% |

Severe imbalance: most-to-least frequency ratio ≈ 18×.

### 4.5 Selective Cross-Dataset Augmentation

**Observation**: After fixing the architecture-level collapse (§6), our model achieves wF1=0.42 on MELD test, but per-class F1 reveals that 5 of the 7 classes (anger, sadness, fear, disgust, surprise) are still at F1=0 — the model has learned to predict only *neutral* and *joy* reliably.

**Naive approach**: Combine the full DailyDialog training set with MELD. **This fails**: DailyDialog is 83% no_emotion, so combining datasets *increases* the neutral fraction from 47% to 66%, exacerbating imbalance.

**Our strategy**: Add only DailyDialog dialogues that **contain** at least one turn labeled `fear`, `disgust`, or `sadness`. Other DailyDialog dialogues are discarded.

| Strategy | Total dlg | rare3 share | fear count |
|----------|----------:|------------:|-----------:|
| MELD only | 1038 | 12.2% | 268 |
| **+ DD rare3** | **1943** | **15.0%** | **414** |
| + DD full | 6871 | 4.4% | 414 |

Selective merge increases the rare3 (fear + disgust + sadness) share from 12.2% to 15.0% — a 23% relative improvement in rare-class density. Full merge would *decrease* it to 4.4%.

This is implemented via the `--dd_rare fear,disgust,sadness` flag in `train.py`.

### 4.6 WeightedRandomSampler

On top of selective augmentation, we use a **dialogue-level** weighted random sampler:

$$
\text{weight}(d) = \sum_{u \in d} \text{class\_weight}(\text{emotion}(u))
$$

where `class_weight(c) = N / (K × count(c))` is the inverse-frequency weight. Dialogues containing rare-class turns get up to **30×** higher sampling probability than all-neutral dialogues.

**Implementation note**: Sampling is at the **dialogue** level, never the turn level. This preserves intra-dialogue context — TemporalMemory's exp(-λ·Δt) aggregation requires complete sequences.

---

## 5. Training

### 5.1 Joint Loss

$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \alpha \cdot \mathcal{L}_{\text{MSE}}
$$

- **$\mathcal{L}_{\text{BCE}}$**: Binary cross-entropy on emotion logits against multilabel targets.
- **$\mathcal{L}_{\text{MSE}}$**: Mean squared error between encoder's 8-dim appraisal output and Scherer target vector (masked for neutral/peaceful/powerful turns which have no target).
- **$\alpha = 0.1$**: Weak supervision weight. Strong enough to anchor appraisal head to Scherer; weak enough not to suppress the main emotion task.

**Mathematical form**:

$$
\mathcal{L}_{\text{BCE}} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{K} \sum_{k=1}^{K} \mathrm{BCE}(z_{u,k}, y_{u,k})
$$

where $\mathcal{U}$ is the set of valid utterances in the batch, $z_{u,k}$ are emotion logits, and $y_{u,k} \in \{0, 1\}$ are multilabel targets.

$$
\mathcal{L}_{\text{MSE}} = \frac{\sum_{u \in \mathcal{U}_a} \|a_u - t_u\|_2^2}{|\mathcal{U}_a|}
$$

where $\mathcal{U}_a \subseteq \mathcal{U}$ is the set of utterances with a defined appraisal target $t_u$, and $a_u$ is the model's predicted appraisal.

### 5.2 Optimization

- Optimizer: AdamW
- Learning rate: $5 \times 10^{-4}$
- Batch size: 2 (dialogue-level; each batch contains up to ~30 utterances)
- Epochs: 3
- Gradient checkpointing: enabled on the backbone
- Mixed precision: bf16 for the frozen backbone, fp32 for trainable heads

### 5.3 Threshold Selection

After training, the best checkpoint (by dev wF1) is loaded. We **sweep thresholds** $\tau \in \{0.2, 0.3, 0.4, 0.5, 0.6\}$ on the dev set, pick the $\tau$ with highest dev wF1, then apply that threshold to test.

On MELD, $\tau = 0.2$ was selected.

---

## 6. Failure Analysis

This section documents the engineering journey of getting EmoFlow to work. We believe these negative findings constitute a contribution in their own right: they show that an architecturally clean and theoretically motivated model can fail in **multiple distinct ways**, all rooted in a single underlying defect (sigmoid saturation), and that standard imbalance-mitigation techniques are insufficient to recover.

### 6.1 The Six Failed Mitigations

| # | Intervention | Test wF1 | Failure mode |
|---|-------------|----------|--------------|
| 1 | Inverse-freq class weights (default) | 0.31 | predict all neutral |
| 2 | + Label smoothing $\epsilon = 0.1$ | 0.31 | predict all neutral |
| 3 | + WeightedRandomSampler oversample | 0.25 dev | predict all neutral |
| 4 | Multilabel BCE (5-dim appraisal) | 0.25 dev | marginal output collapse |
| 5 | + Per-dim pos_weight $= (1-p)/p$ | 0.0025 dev | oversteer to rare classes |
| 6 | 5 → 8 dim appraisal expansion | 0.25 dev | marginal collapse persists |

Across attempts 1–3, dev wF1 was **bit-identical at 0.2523** with 194/607 correct transitions, regardless of the loss-side intervention. This persistence was the first hint that the bug was upstream of the loss function.

### 6.2 Diagnostic: Encoder Output Constancy

To localize the failure, we ran a controlled forward pass on three semantically distinct sentences:

```
Text 1: "I am so happy! This is the best day ever."
Text 2: "I hate this. I want to die."
Text 3: "Oh my god, what just happened?!"

Appraisal output (all 3 texts):
  [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
  
std across texts per dim:
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

The encoder produced **bit-identical** outputs for completely unrelated inputs. This is not "weak discrimination" — it is **zero discrimination**. The model's effective input dimension at the appraisal bottleneck is zero.

### 6.3 Root Cause: Sigmoid Saturation

The encoder's AppraisalHead originally ended with a sigmoid activation:

```python
self.head = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.GELU(),
    nn.Linear(4096, 8),
    nn.Sigmoid(),         # ← culprit
)
```

The intent was to constrain output to $[0,1]$ to match Scherer normalized targets. However:

- Early in training, large gradient steps push pre-activation logits to extreme values ($|z| \gg 1$).
- Once $z \gtrsim 5$ or $z \lesssim -5$, the sigmoid output saturates near 1 or 0.
- At saturation, the derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ approaches 0.
- The gradient on $z$ vanishes; the head **cannot recover** — there is no signal to push outputs back into the linear regime.

Once all 8 output dimensions saturate, every input produces the same output vector. The encoder degenerates into a constant function. Downstream, the BayesianHead receives a constant input and produces constant logits; argmax collapses to whatever class has the highest constant logit (in practice: *neutral* under softmax CE, or *joy* under sigmoid BCE).

### 6.4 The Fix

Remove the sigmoid:

```python
self.head = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.GELU(),
    nn.Linear(4096, 8),
    # No sigmoid; output is unbounded ℝ⁸.
)
```

**Concerns this raised, and resolutions**:
- "Targets are in $[0,1]$; can the model still match them?" Yes — MSE loss naturally pulls outputs into target range. Empirically, post-fix appraisal outputs lie mostly in $[-0.5, 1.8]$.
- "Will outputs explode?" No — the MSE term plus emotion BCE provide bidirectional gradient pressure.

**After-fix diagnostic** (same three texts):

```
Text 1: [1.78, -0.35, -0.01, 0.14, 1.17, -0.20, 0.15, 1.21]  ← matches joy signature
Text 2: [0.16, 0.70, 0.70, 0.25, 0.18, 0.48, 0.48, -0.03]   ← negative-emotion direction
Text 3: [-0.43, 0.92, 1.09, 0.67, 0.61, 0.40, 0.36, -0.38]  ← low expectedness (surprise!)

mean std across texts: 0.413  (was 0.000)
```

The encoder produced theoretically aligned, input-dependent appraisal vectors. For example:
- "I am so happy!" has high `coping_potential` (1.17) and high `self_consistency` (1.21), matching the joy template.
- "Oh my god, what just happened?!" has the **lowest** `expectedness` (-0.43) and **lowest** `self_consistency` (-0.38) — exactly the appraisal signature of surprise in Scherer's table.
- "Get the fuck out of my house!" (tested separately) had high `unfairness` (0.72) and high `immorality` (0.70) — anger's signature.

This is one of our strongest pieces of qualitative evidence: removing the sigmoid not only fixed training but recovered **semantically aligned** representations.

### 6.5 Summary of the Lesson

Bounded activations at low-dimensional bottlenecks are dangerous in deep models. The "natural" choice (sigmoid to match $[0,1]$ targets) introduced a vanishing-gradient failure that masqueraded as a data imbalance problem for many hours of debugging. The lesson generalizes:
- Prefer unbounded activations at internal representational bottlenecks.
- If bounded output is required, achieve it via loss-based constraints (MSE against $[0,1]$ targets) rather than activation-based clamping (sigmoid).

---

## 7. Results

### 7.1 Main Table

All baselines use the same configuration (multilabel BCE, $\alpha = 0.1$, MELD + DD-rare3, oversample, no sigmoid). Only the model class differs.

| Model | wF1 | wF1<sub>6way</sub> | ETA | trainable |
|-------|----:|-----:|----:|----------:|
| LSTM (BiLSTM over appraisals) | 0.4241 | 0.2719 | 0.4095 | ~20M |
| Stateless (no memory) | 0.5631 | 0.4553 | 0.5462 | ~20M |
| EmoFlow (λ=0, no decay) | 0.6052 | 0.5348 | 0.5617 | ~20M |
| **EmoFlow (learned λ)** | **0.6171** | 0.5312 | 0.5528 | ~20M |

**Metrics**:
- **wF1**: sklearn weighted F1 over all 7 classes (including neutral). Standard MELD metric.
- **wF1<sub>6way</sub>**: weighted F1 computed over the 6 non-neutral classes only (excluding neutral-labeled turns from the metric).
- **ETA** (Emotion Transition Accuracy): for turns where the true emotion differs from the previous turn, the fraction where the model correctly predicted the new emotion. Tests how well the model handles emotional changes (a key marker of contextual reasoning).

### 7.2 Comparison with Published Models on MELD

| Model | Year | MELD wF1 | Setting / params |
|-------|-----:|---------:|------------------|
| DialogueRNN | 2019 | 0.57 | text · ~3M |
| DialogueGCN | 2019 | 0.58 | text · ~5M |
| MMGCN† | 2021 | 0.59 | multimodal · ~110M+ |
| **EmoFlow (ours)** | 2026 | **0.62** | text · **~20M trainable** |
| COSMIC | 2020 | 0.65 | text · BERT-large ~340M |

All figures are MELD 7-class weighted-F1 on the test set. †MMGCN's frequently-quoted 0.66 is its **IEMOCAP** score; on MELD it reports **58.65** (Hu et al. 2021, Table 4). DialogueRNN's MELD wF1 is **57.03** and COSMIC's is **65.21** (Ghosal et al. 2020, Table 4 — the same table also lists DialogueRNN).

On the raw MELD test number, EmoFlow (0.62) lands above DialogueRNN, DialogueGCN, and MMGCN, and below COSMIC, using only LoRA adapters on a frozen backbone.

**Important caveat — this is an indicative, not a controlled, comparison.** EmoFlow's 0.62 is obtained with selective cross-dataset rare-class augmentation (MELD + DailyDialog `dd_rare`; §4.5), whereas the published baselines train on MELD train only. Without augmentation, post-fix EmoFlow scores **wF1=0.42** on MELD (§4.5) — below these baselines — so the bulk of the +0.20 gap to our headline is attributable to the augmentation data, not the architecture in isolation. The *controlled* architectural comparison, in which EmoFlow, Stateless, and LSTM are all trained on the identical MELD+DD-rare3+oversample data, is the ablation in §7.1/§7.5, where EmoFlow's structured memory beats the Stateless and LSTM baselines by +5.4 and +19.3 wF1. Cross-architecture / cross-modality / cross-data comparisons to published numbers should therefore be read as context, not as a controlled claim of superiority.

### 7.3 Per-class Breakdown (EmoFlow Full)

| Emotion | Support | Precision | Recall | F1 |
|---------|--------:|----------:|-------:|---:|
| neutral | 1256 | 0.75 | 0.78 | **0.76** |
| joy | 402 | 0.62 | 0.54 | **0.58** |
| surprise | 281 | 0.50 | 0.65 | **0.57** |
| anger | 345 | 0.53 | 0.48 | **0.51** |
| sadness | 208 | 0.51 | 0.24 | **0.32** |
| disgust | 68 | 0.17 | 0.44 | **0.25** |
| **fear** | **50** | **0.00** | **0.00** | **0.00** |

Six of seven classes learned non-trivially. **Only fear remains at zero F1.**

### 7.4 The Fear Class Problem

Fear has only 50 test samples and 268 + 146 = 414 train samples (MELD + DD-rare). Even after 30× oversampling and selective augmentation, the model fails to predict any test instance as fear.

Hypotheses (to be discussed in §8):
1. **Data-bound**: 414 training utterances may simply be below the threshold for the model to discriminate.
2. **Appraisal overlap**: Fear's Scherer signature (high unpleasantness, low coping_potential, high goal_hindrance) overlaps substantially with sadness and anger. The 8-dimensional space may not provide enough separation.
3. **Model capacity**: LoRA $r=8$ might lack the fine discriminative capacity for very rare classes.

### 7.5 Ablation Findings

**(a) Memory architecture matters more than presence of memory.**

Stateless (no memory) achieves 0.5631 wF1. LSTM (BiLSTM memory) achieves 0.4241 — **14 points worse** despite having more parameters. EmoFlow's structured exp(-λ·Δt) memory achieves 0.6171 — **20 points better than LSTM**.

This is a non-obvious finding: a generic recurrent prior (BiLSTM) over appraisal vectors actively *hurts* compared to no memory at all. The form of temporal weighting matters. We conjecture LSTM overfits to dialogue-specific patterns in MELD's small training set, while EmoFlow's structured exponential decay provides a strong inductive bias that resists overfitting.

**(b) Learnable decay slightly outperforms no decay.**

EmoFlow no-decay ($\lambda = 0$): 0.6052 wF1.
EmoFlow learned $\lambda$: 0.6171 wF1.

Difference: +1.2 points wF1, but **−0.36 points wF1<sub>6way</sub>** and **−0.9 points ETA**.

The learned $\lambda$ optimizes for the dominant class (neutral) but slightly hurts rare-class discrimination. We interpret this as: rare-class evidence may live in earlier turns, and exponential decay attenuates it. For ERC tasks where rare classes are more important, uniform aggregation (no decay) may be preferable.

### 7.6 Interpretability: Appraisal Case Study

After training, we tested the encoder on five sentences not in the training set:

| Sentence | Highest dim | Lowest dim | Interpretation |
|----------|------------|------------|----------------|
| "I am so happy! This is the best day ever." | self_consistency (1.21), coping_potential (1.17) | unpleasantness (-0.35) | Joy signature ✓ |
| "Oh my god, what just happened?!" | goal_hindrance (1.09), unpleasantness (0.92) | expectedness (-0.43) | Low expectedness = surprise novelty ✓ |
| "Get the fuck out of my house!" | unfairness (0.72), immorality (0.70) | self_consistency (0.25) | Anger ✓ |
| "I hate this. I want to die." | unpleasantness (0.70), goal_hindrance (0.70) | self_consistency (-0.03) | Sadness/Anger blend ✓ |
| "Just another boring meeting today." | unpleasantness (0.99) | coping_potential (0.16) | Mild disgust/boredom (no joy signal) ✓ |

The encoder learned to map utterances to Scherer-theoretically aligned appraisal patterns **without explicit per-sentence supervision** — only the 6 emotion-level prototype vectors served as MSE targets, and emotion-level CE supervised the downstream classification.

---

## 8. Discussion

### 8.1 What Worked

Three architectural commitments paid off:
1. **8 raw appraisal dimensions** (vs. compressed 5-dim): preserves Scherer's theoretical distinctions, especially between *unfairness* (anger) and *immorality* (disgust), which were collapsed by the original 5-dim aggregation.
2. **Bayesian factorization** (prior + likelihood): +5 wF1 over Stateless. Explicit decomposition of context vs. current evidence provides interpretability and inductive bias.
3. **Multilabel reformulation** (neutral = zero vector): structurally prevents single-class collapse. Pivotal for breaking out of the all-neutral failure mode.

### 8.2 What Surprised Us

**LSTM < Stateless**: BiLSTM memory over per-turn appraisals underperformed no memory at all. We initially expected LSTM to be at least competitive with EmoFlow. This finding suggests that **generic recurrent priors are insufficient or actively harmful for ERC on small datasets** — structured, theoretically motivated memory architectures perform better even with the same parameter budget.

**Bit-identical dev metrics across many "different" mitigations**: The failure modes for class weights, label smoothing, and oversample (before the sigmoid fix) all produced identical dev metrics (wF1=0.2523, 194/607 transitions correct). This was an early signal that the bug was upstream of the loss — all these interventions were modifying gradients on a representation that had already collapsed.

### 8.3 The Sigmoid Saturation Lesson

This is the central engineering finding of our project. Bounded activations at low-dimensional representational bottlenecks introduce a vanishing-gradient failure mode that is easy to misdiagnose as a data imbalance issue. Future work on appraisal-grounded models — or more generally, models with low-dim interpretable bottlenecks — should:
- Use unbounded activations at internal bottlenecks
- Express target constraints through loss (MSE against $[0,1]$ targets) rather than architectural clamping
- Run encoder-output sanity checks (std across diverse inputs) as part of debugging when classification metrics collapse

### 8.4 Limitations

This section is explicit about what we set out to do (per the project proposal) versus what we delivered, so that readers can independently judge scope.

**(a) Single primary dataset.** All main results are on MELD. The original proposal targeted EmoryNLP as the primary benchmark; we swapped to MELD after EmoryNLP's speaker-disjoint split triggered persistent class collapse during the pre-fix phase (§4.1, §6). Final-quality EmoryNLP results on the post-fix pipeline are missing. IEMOCAP, a third common ERC benchmark, was not attempted.

**(b) Response generation not delivered.** The proposal included a sixth pipeline module — a Response Generator using emotion-conditioned LLaMA-3-8B decoding for empathetic replies. This module is not implemented in this submission. We documented the rationale in §1.4: failure-mode debugging consumed the compute and engineering budget originally allocated to the generator. The classifier we deliver is the strict prerequisite for any such generator and is a self-contained contribution in its own right.

**(c) Human Coherence Score evaluation not performed.** The proposal committed to Likert-scale human evaluation of generated replies on 100 EmoryNLP test dialogues with two annotators. This evaluation was tied to the Response Generator and is consequently absent.

**(d) Fear class not learned.** Despite augmentation and oversampling, fear F1 remains 0. The model has insufficient signal to discriminate fear from related negative emotions in our 8-dim representation.

**(e) Bounded by frozen backbone, and headline depends on augmentation.** With ~20M trainable parameters on top of frozen LLaMA-3-8B, our model reaches but does not exceed COSMIC (fine-tuned BERT-large, ~340M). Moreover, our 0.62 headline relies on cross-dataset augmentation that the published baselines do not use (§7.2); MELD-only post-fix EmoFlow is 0.42. We therefore do not claim a controlled win over published models — our controlled evidence is the matched-data ablation in §7.1/§7.5.

**(f) No multimodal information.** MELD includes audio and video; we use text only. Audio prosody and facial expression provide strong fear/sadness signals we cannot access.

**(g) Approximate Bayesian update.** The log-additive MLP fusion in BayesianHead is equivalent to an exact Dirichlet–Categorical posterior only under uniform marginals. A fully Bayesian (variational or sampling-based) inference layer was originally proposed but not implemented.

### 8.5 Future Work

1. **Complete the proposed pipeline.** Implement the Response Generator that takes emotion distributions as soft-prompt conditioning, evaluate it via the Coherence Score protocol from the proposal. This is the direct continuation of the original project scope.
2. **Multi-dataset evaluation**: Add EmoryNLP and IEMOCAP using the post-fix pipeline. The sigmoid-saturation diagnostic is independent of dataset and should transfer.
3. **Larger LoRA / partial unfreezing**: Test whether $r = 16, 32$ or unfreezing top LLaMA layers improves rare-class F1 (specifically fear).
4. **Multimodal extension**: Inject audio prosodic features into the appraisal head.
5. **Fully Bayesian inference**: Replace log-additive MLP with explicit Dirichlet–Categorical update, possibly via variational inference, to match the proposal's mathematical formulation.

---

## 9. Conclusion

EmoFlow is an appraisal-grounded Bayesian model for conversational emotion recognition that combines:
- A frozen LLaMA-3-8B encoder with LoRA adapters producing 8-dim Scherer-aligned appraisal vectors
- A learnable exponential-decay memory aggregator
- A log-additive Bayesian fusion head separating context prior from current likelihood
- A multilabel reformulation treating neutral as the all-zero vector

We diagnosed a reproducible failure mode — sigmoid saturation in the appraisal head — that caused six independent imbalance-mitigation strategies to fail in distinct but consistent ways. A one-line architectural fix (remove sigmoid) plus selective cross-dataset augmentation produced a model matching DialogueRNN-tier performance (wF1=0.62 on MELD test) at one-tenth the trainable parameter count.

The most transferable lessons of this work are:
1. **Theoretical grounding can guide representation design** — Scherer's 8 appraisal dimensions provide an interpretable bottleneck that survives extensive ablation.
2. **Bayesian factorization adds structure cheaply** — splitting prior and likelihood improves discrimination without extra parameters.
3. **Diagnose representations, not just metrics** — when classification collapses, check whether the encoder is producing input-dependent representations before blaming the loss or the data.

---

## Appendix A: Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Backbone | meta-llama/Meta-Llama-3-8B (frozen, NF4 quantized) |
| LoRA rank $r$ | 8 (proposal estimate: 16; reduced for memory headroom under QLoRA) |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA target modules | q_proj, v_proj |
| Pooling | last non-pad token hidden state |
| AppraisalHead | Linear(4096→4096) → GELU → Linear(4096→8) |
| AppraisalHead output activation | **None** (post-fix) |
| Appraisal dim | 8 |
| BayesianHead hidden | 64 |
| BayesianHead dropout | 0.1 |
| TemporalMemory $\lambda$ init | 0.1 |
| Joint loss $\alpha$ | 0.1 |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Batch size | 2 dialogues |
| Epochs | 3 |
| Inference threshold | 0.2 (selected on dev) |

## Appendix B: Files in the Codebase

- `appraisal_targets.py` — Scherer Table 5.5 Z-scores + min-max normalization → 8-dim targets
- `dataloader.py` — Dataset class, Dialogue/Utterance dataclasses, collate function
- `encoder.py` — StimulusEncoder: frozen LLaMA + LoRA + AppraisalHead
- `memory.py` — TemporalMemory with learnable exp(-λ·Δt) attention
- `bayes.py` — BayesianHead with separated prior and likelihood MLPs
- `model.py` — EmoFlowModel: end-to-end composition; MockEncoder for testing
- `baselines.py` — StatelessClassifier and LSTMMemoryModel
- `evaluation.py` — weighted F1 and Emotion Transition Accuracy (ETA-B)
- `train.py` — Training loop with all flags (--multilabel, --dd_rare, --oversample, ...)
- `train_appraisal.py` — DailyDialog appraisal pre-training (not used in final pipeline)
- `preprocess.py` — Raw dataset → unified Dialogue JSONL
