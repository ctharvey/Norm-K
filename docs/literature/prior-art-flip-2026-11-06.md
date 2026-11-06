# Prior-art audit: frozen-embedding negation flip and Norm-K

**Reviewed:** 2026-11-06  
**Scope:** literature explicitly discussed in the Norm-K research thread, the user's uploaded Schumann–Marx Gómez (2023) PDF, Cui et al. (2020), plus close methodological competitors.  
**Project evidence:** [v4 placebo experiment](../../reports/flip-placebo-v4.md), [frozen v4 protocol](../../docs/preregistration-flip-placebo-v4.md), [v2 public replication](../../reports/public-fair-k-v2.md).  
**Machine-readable source index:** [sources.tsv](sources.tsv).  

## Executive decision — narrow continuation, park speculative extensions

**Do not claim novelty for:** using cosine in NLI, controlled text/sentence transformations, inference-time editing, composing logical cosine scores, negate-sensitive retrieval, trained contradiction operators, token-level alignment, or fact decomposition. The strongest direct precedents are Zhu et al. 2018, Cui et al. 2020, Schumann–Marx Gómez 2023, Faltings et al. 2025, MorphNLI 2025, DEO 2026 and E-SENS 2026. Their mathematical and training assumptions differ from the current Norm-K v4 score, but the *general* idea is heavily preempted.

**Possible research space, unproven:** the *specific fixed, symmetric, bidirectional negation-response score* of two **independently embedded** statements, evaluated without task-trained parameters:

```text
F_N(A,B) = 0.5 * [ cos(E(N(A)),E(B)) + cos(E(A),E(N(B))) ] - cos(E(A),E(B))
```

The inspected papers do **not establish exact prior publication** of this formula; this is **not** proof of novelty. The value would lie in a robust causal/held-out result, not in merely writing a new combination of cosine values.

**What v4 has actually shown:** on a new but still MultiNLI is/was cohort (1,500 pairs), negation flip beat base cosine by +0.296 AP (OpenAI text-embedding-3-small) and +0.224 AP (MPNet). Against the best of two small neutral edits it was +0.288 and +0.117. The MPNet model has MultiNLI training exposure, and its cleaner WANLI negation-vs-placebo result is inconclusive. Dedicated NLI remains far better on WANLI. These are ranking and intervention observations; neither universal semantics nor practical superiority is established.

**Main remaining threats:** negation produces far larger geometric displacement than the tested placebos; shuffled-pair AP stays ~0.44–0.45 versus 0.327 prevalence; restrictive is/was grammar and MultiNLI author artifacts; cross-encoder superiority; unmeasured actual cost; narrow coverage and no domain-real memory examples.

### Mathematical interpretation worth testing before new models

For unit vectors a,b and their negated embeddings a_N,b_N, let d_a=a_N-a, d_b=b_N-b, m_a=1-cos(a,a_N) and m_b=1-cos(b,b_N). Algebraically:

```text
F_N = 0.5*(d_a · b + a · d_b)
    = 0.5*((d_a - d_b) · (b - a)) - 0.5*(m_a+m_b)
```

Thus the score includes a pair-dependent term and a two-sentence individual edit-magnitude term, consistent with why partner shuffling is informative. This is an exact identity, not a new empirical discovery. Splitting these requires no new embedding calls if the vectors are cached. Note that the decomposition may be algebraically redundant with existing cosine features; compare on sealed pairs and avoid retrospective optimization.

### Two-step confirmatory gate (proposed; **not yet preregistered**)

1. **Displacement-matched placebo and pair-artifact control:** choose non-negating edits that preserve target meaning while matching the *distribution* of cosine displacement and grammatical position of negation, using a label-blind pilot; human-audit validity, including original negation scope. Score fresh independent pairs with the same fixed flip. Include original vs neutral vs non-negating-semantic edit, statement-only edit magnitude, shuffled partner, swapped pair order, and a simple lexical/hash control. Freeze dataset selection, primary AP delta, intervals, and direction before embedding new pairs.
2. **Truly independent task/domain and operational check:** test NaN-NLI/sub-clausal negation and appropriately licensed independent factual/atomic-belief cases (not just MultiNLI is/was); retain a strong off-the-shelf NLI baseline, report AP plus recall at a fixed review budget, latency, coverage and measured token/API cost. Avoid transferring labels from NLI contradiction to Menhir same-subject/same-time exclusive-slot conflict without an explicit mapping.

**Candidate precommitted stop rule, subject to actual preregistration:** require delta AP at least +0.03 over the *strongest valid displacement-matched/statement-only control*, bootstrap lower CI >0, independent OOD replication without a sign flip, and a practical improvement in review-cost/recall relative to an NLI baseline. A significant semantic advantage without a practical advantage may still support a qualified research note, not integration. If the paired causal/OOD gains disappear, **pin the research with negative results and stop spending on K, asinh, random projections, extra encoders and chunking variants**. Preserve a short reproducible report and focus Menhir on typed assertions, valid time and NLI where warranted.

**Park now regardless:** original four-point K as a candidate detector (v2 failed to replicate), signed log/asinh, per-token/chunk-level flip, learned projections, and extra transformation families. Do not treat 'not yet disproved' as budget to keep feature-searching indefinitely.

## Source verification and treatment

Each entry below gives **what it actually does**, **where it overlaps**, and **what to do about it**. `PDF method inspected` means at least the relevant method/evaluation passages were checked, not a line-by-line replication; `abstract checked` means the judgment is provisional. Bibliographic links point to publishers/ACL Anthology/arXiv/OpenReview. **No third-party PDFs are redistributed in this repository** pending license verification. The user supplied Schumann–Marx Gómez's six-page PDF in chat; its canonical DOI and the authors' [repository](https://github.com/schumanng/unsupervised-contradiction-detection) are recorded. The attachment's local bytes were not silently copied to GitHub.

## A. Closest scoring, edit-at-inference, and direct contradiction precedents

### P01. Exploring Semantic Properties of Sentence Embeddings (2018)

**Authors:** Zhu, Li & de Melo. **Source:** [publication/preprint](https://aclanthology.org/P18-2100/), [source PDF](https://aclanthology.org/P18-2100.pdf). **Evidence depth:** PDF method inspected.

- **Method:** Creates sentence triplets with controlled negation, synonym substitutions, quantifiers, clause edits and other transformations, then tests inequalities between embedding cosine scores.
- **Overlap / novelty consequence:** Direct antecedent for measuring the response of frozen embeddings to edits. It diagnoses representation quality rather than turning a bidirectional edit response of an arbitrary A/B pair into a contradiction-ranking score.
- **Norm-K action:** Reproduce its negation-detection triplet condition alongside flip; do not call controlled cosine sensitivity new.

### P02. Unsupervised Natural Language Inference via Decoupled Multimodal Contrastive Learning (2020)

**Authors:** Cui, Zheng & Wang. **Source:** [publication/preprint](https://aclanthology.org/2020.emnlp-main.444/), [source PDF](https://aclanthology.org/2020.emnlp-main.444.pdf). **Evidence depth:** PDF method and evaluation inspected.

- **Method:** Contrastively pretrains a decoupled text encoder against images; at NLI inference uses cosine between the two *unmodified* sentence vectors with two thresholds (entailment above ψ1, contradiction below ψ2, neutral otherwise).
- **Overlap / novelty consequence:** Direct precedent for parameter-free inference-time cosine-based NLI classification *after encoder training*, not test-time editing, four-vector comparison, or frozen generic encoder selection.
- **Norm-K action:** Cite §2.2 and §4.2, report simple threshold-cosine baseline, and distinguish no NLI-label training from no encoder training. Original paper confirms the user's observation.

### P03. Unsupervised Natural Language Inference Using PHL Triplet Generation (2021)

**Authors:** Varshney et al.. **Source:** [publication/preprint](https://arxiv.org/abs/2110.08438), [source PDF](https://arxiv.org/pdf/2110.08438). **Evidence depth:** Primary abstract and method examples checked.

- **Method:** Generates labeled premise–hypothesis examples by rule-based transformations then trains an NLI model on those automatically generated labels.
- **Overlap / novelty consequence:** Establishes transformations for synthetic supervision; does not use a training-free cosine flip at inference.
- **Norm-K action:** Reference as the direct ancestor of Schumann–Marx Gómez, and audit synthetic-label shortcuts.

### P04. Unsupervised Contradiction Detection using Sentence Transformations (2023)

**Authors:** Schumann & Marx Gómez. **Source:** [publication/preprint](https://doi.org/10.1109/ICNLP58431.2023.00065) (user-uploaded PDF also inspected). **Evidence depth:** User-supplied six-page PDF read.

- **Method:** Builds automatically labeled sentence pairs via negation, paraphrases, antonyms and other transformations; fine-tunes bert-base-uncased sequence classifier on ~300k examples. Reports 71.42% SNLI test accuracy on selected transforms and 93.32% for supervised comparison.
- **Overlap / novelty consequence:** Similar vocabulary and synthetic transformations, categorically different inference: trained BERT classifier instead of zero-shot four-embedding response. Its accuracy must not be compared directly to Norm-K AP.
- **Norm-K action:** Cite its §V.C training method and §VI; inspect transformation-noise and out-of-domain findings. Uploaded PDF is not redistributed in repository without rights confirmation.

### P05. Enhancing Retrieval Systems with Inference-Time Logical Reasoning (2025)

**Authors:** Faltings, Wei & Bao. **Source:** [publication/preprint](https://aclanthology.org/2025.acl-short.34/), [source PDF](https://aclanthology.org/2025.acl-short.34.pdf). **Evidence depth:** PDF introduction and methodology inspected.

- **Method:** Parses negation/conjunction/disjunction into logical query components and composes their frozen-embedding cosine matching scores at retrieval time.
- **Overlap / novelty consequence:** Strong proof that inference-time logical score composition without encoder updates is established. It does not test Norm-K's symmetric A↔B negation flip for contradictions.
- **Norm-K action:** Treat as highest-priority architectural prior art, explicitly compare its NOT scoring rule, cost, and denominator/matching conventions to flip.

### P06. MorphNLI: A Stepwise Approach to Natural Language Inference Using Text Morphing (2025)

**Authors:** Negru et al.. **Source:** [publication/preprint](https://aclanthology.org/2025.findings-naacl.385/), [source PDF](https://aclanthology.org/2025.findings-naacl.385.pdf). **Evidence depth:** Primary paper method passages inspected.

- **Method:** Generates atomic edits moving premise toward hypothesis at inference time, runs an off-the-shelf NLI model on successive steps, and aggregates labels; reports cross-domain benefits.
- **Overlap / novelty consequence:** Negates novelty of inference-time editing for NLI generally. Distinct from a fixed lexical negation edit read through a frozen embedding similarity function without an NLI model.
- **Norm-K action:** Use as edit-based and clause/atomic-step comparator; do not claim test-time edit-based NLI is new.

### P07. DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval (2026)

**Authors:** Lee et al.. **Source:** [publication/preprint](https://arxiv.org/abs/2603.09185), [source PDF](https://arxiv.org/pdf/2603.09185). **Evidence depth:** Preprint PDF introduction/method inspected.

- **Method:** Separates positive and negative parts of a retrieval query and optimizes its representation with a contrastive objective without updating the encoder.
- **Overlap / novelty consequence:** A close training-free, negation-aware manipulation of embedding scores, but it solves exclusion retrieval and optimizes query geometry rather than measuring symmetric contradiction flip.
- **Norm-K action:** Benchmark inference costs and compare against exclusion-style penalty controls where task semantics permit; preprint publication status not established.

### P08. E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval (2026)

**Authors:** Kim, Myung & Han. **Source:** [publication/preprint](https://aclanthology.org/2026.groundlm-1.28/), [source PDF](https://aclanthology.org/2026.groundlm-1.28.pdf). **Evidence depth:** Primary abstract and method summary checked.

- **Method:** Training-free reranker constructs a trap/excluded-concept query and subtracts its similarity from the original matching score.
- **Overlap / novelty consequence:** Subtractive composition of pretrained similarities is established; this is exclusion retrieval, not generic contradiction of two claims.
- **Norm-K action:** Use a cheap subtraction baseline and discuss differences in task and false positives.

### P09. Learning Entailment-Based Sentence Embeddings from Natural Language Inference (2019)

**Authors:** Karimi Mahabadi, Mai & Henderson (attributed; confirm final record). **Source:** [publication/preprint](https://openreview.net/forum?id=BkxackSKvH), [source PDF](https://openreview.net/pdf?id=BkxackSKvH). **Evidence depth:** OpenReview PDF inspected; under review ICLR 2020.

- **Method:** Defines parameter-free entailment/contradiction interaction functions but *trains the sentence encoder* with those functions as inductive bias.
- **Overlap / novelty consequence:** Predefined vector contradiction operators are not new. It does not imply those operators work on frozen general-purpose embeddings.
- **Norm-K action:** Treat as preprint/review submission until bibliographic status confirmed; compare algebra rather than only titles.

### P10. Gated Residual Discrepancy for Efficient Bi-Encoder Factuality Verification (2026)

**Authors:** Liu et al.. **Source:** [publication/preprint](https://aclanthology.org/2026.emnlp-main.84/), [source PDF](https://aclanthology.org/2026.emnlp-main.84.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Trains or adapts a bi-encoder factuality system with dual-view vector differences and adaptive gating to avoid contextual-noise masking; reports strong speed relative to cross-encoding.
- **Overlap / novelty consequence:** Close competing *engineering* direction for fact contradiction/factuality in embedding geometry; not the same parameter-free edit response.
- **Norm-K action:** If Menhir operational filtering remains the goal, compare GRD cost/recall before continuing bespoke geometry.

## B. Negation behavior, transformations, and benchmark design

### P11. OssCSE: Overcoming Surface Structure Bias in Contrastive Learning for Unsupervised Sentence Embedding (2023)

**Authors:** Shi et al.. **Source:** [publication/preprint](https://aclanthology.org/2023.emnlp-main.448/), [source PDF](https://aclanthology.org/2023.emnlp-main.448.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Shows embedding similarity can favor near-identical negated sentences over syntactically different paraphrases and proposes training strategies to counter surface-form bias.
- **Overlap / novelty consequence:** Validates the problem and the inadequacy of edit-word controls; not a zero-shot contradiction score.
- **Norm-K action:** Add large meaning-preserving paraphrases and lexical-overlap matching to edit controls.

### P12. SNCSE: Contrastive Learning for Unsupervised Sentence Embedding with Soft Negative Samples (2022)

**Authors:** Wang et al.. **Source:** [publication/preprint](https://arxiv.org/abs/2201.05979), [source PDF](https://arxiv.org/pdf/2201.05979). **Evidence depth:** Primary abstract checked.

- **Method:** Uses negated sentences as soft negative samples within contrastive learning and a bidirectional margin loss to train embeddings.
- **Overlap / novelty consequence:** Shows negation response can be deliberately learned, weakening claims of a universal frozen-space phenomenon.
- **Norm-K action:** Compare encoder training provenance and test unseen models trained with and without such objectives.

### P13. ALIGN-SIM: A Task-Free Test Bed for Evaluating and Interpreting Sentence Embeddings through Semantic Similarity Alignment (2024)

**Authors:** Mahajan et al.. **Source:** [publication/preprint](https://aclanthology.org/2024.findings-emnlp.436/), [source PDF](https://aclanthology.org/2024.findings-emnlp.436.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Evaluates 13 encoders on five tests: semantic distinction, synonym and antonym replacement, non-negated paraphrase, and word jumbling.
- **Overlap / novelty consequence:** Controlled edit sensitivity and frozen-embedding diagnosis are established, but not an A/B contradiction ranking from a directional flip.
- **Norm-K action:** Reuse its criteria as edit-strength and meaning-preservation diagnostics.

### P14. NevIR: Negation in Neural Information Retrieval (2024)

**Authors:** Weller, Lawrie & Van Durme. **Source:** [publication/preprint](https://aclanthology.org/2024.eacl-long.139/), [source PDF](https://aclanthology.org/2024.eacl-long.139.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Constructs retrieval document pairs differentiated by critical negation; compares bi-encoders, late-interaction and cross-encoders.
- **Overlap / novelty consequence:** Direct external stress test for negation sensitivity; not a paired-claim contradiction ranking and not our scoring formula.
- **Norm-K action:** Include as separate task-specific evaluation, not fold its retrieval metric into MultiNLI AP.

### P15. Not another Negation Benchmark: The NaN-NLI Test Suite for Sub-clausal Negation (2022)

**Authors:** Truong et al.. **Source:** [publication/preprint](https://aclanthology.org/2022.aacl-main.65/), [source PDF](https://aclanthology.org/2022.aacl-main.65.pdf). **Evidence depth:** Primary abstract and examples checked.

- **Method:** Linguistically curated sub-clausal negation NLI pairs with targeted edits and annotated scope/negation type.
- **Overlap / novelty consequence:** A direct threat to v4's very narrow is/was grammatical intervention; failures of scope can masquerade as successful opposite-polarity edits.
- **Norm-K action:** Use as an untouched out-of-distribution kill set and report scope errors separately.

### P16. A Comprehensive Taxonomy of Negation for NLP and Neural Retrievers (2025)

**Authors:** Petcu et al.. **Source:** [publication/preprint](https://aclanthology.org/2025.findings-emnlp.839/), [source PDF](https://aclanthology.org/2025.findings-emnlp.839.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Defines logical/linguistic negation types and builds balanced evaluation sets for retrievers.
- **Overlap / novelty consequence:** Supplies taxonomy and balanced sampling, not a zero-shot cosine flip.
- **Norm-K action:** Freeze type-stratified tests rather than letting one easily negatable verb form dominate.

### P17. Polarity inversion operators in PLM (2025)

**Authors:** Kletz, Amsili & Candito. **Source:** [publication/preprint](https://aclanthology.org/2025.conll-1.20/), [source PDF](https://aclanthology.org/2025.conll-1.20.pdf). **Evidence depth:** PDF introduction and abstract inspected.

- **Method:** Trains learned operators on contextual verb-token embeddings to invert polarity; involution/double application is imperfect.
- **Overlap / novelty consequence:** Demonstrates a polarity operator can be approximated but is not necessarily algebraic; operates token-level and learns mapping unlike Norm-K.
- **Norm-K action:** Check repeated-negation, different verb scopes and encoder response asymmetry; don't assume edit(N(edit(s))) returns vector s.

### P18. Strong hallucinations from negation and how to fix them (2024)

**Authors:** Asher & Bhar. **Source:** [publication/preprint](https://aclanthology.org/2024.findings-acl.752/), [source PDF](https://aclanthology.org/2024.findings-acl.752.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Models negation as an operation constraining an LM's internal representations and evaluates NLI/cloze effects.
- **Overlap / novelty consequence:** Negation as a representation-space operation is established; applies within model inference rather than separately embedded string edits.
- **Norm-K action:** Keep operator semantics separate from embedding metric observations.

### P19. Computation of sentence similarity score through hybrid deep learning with a special focus on negation sentence (2026)

**Authors:** Rohit M et al.. **Source:** [publication/preprint](https://www.nature.com/articles/s41598-025-34084-2), [source PDF](https://www.nature.com/articles/s41598-025-34084-2_reference.pdf). **Evidence depth:** Full open article methodology inspected.

- **Method:** NAS Scorer combines multiple embeddings, BiLSTM and explicit negation-count/parity/position features in a supervised similarity regressor; uses an annotated NSSD set.
- **Overlap / novelty consequence:** Negation-aware composite similarity predates Norm-K, but this is a trained model relying on explicit lexical features, not a fixed untrained four-vector test-time score.
- **Norm-K action:** Correct earlier shorthand: 'Negation-Aligned Similarity Scorer' names the method, not the paper. Consider negation parity/position controls.

### P20. SemAntoNeg: It Is Not Easy To Detect Paraphrases (2022)

**Authors:** Vahtola, Creutz & Tiedemann. **Source:** [publication/preprint](https://aclanthology.org/2022.blackboxnlp-1.20/), [source PDF](https://aclanthology.org/2022.blackboxnlp-1.20.pdf). **Evidence depth:** Paper evaluation extract inspected.

- **Method:** Tests frozen language-model embeddings on negated-antonym paraphrase pairs; 3,152 examples; evaluates whether correct paraphrases have higher cosine.
- **Overlap / novelty consequence:** Direct relevant nontrivial negation/paraphrase generalization and cosine ranking, not contradiction detector with edited A/B vectors.
- **Norm-K action:** Add as a separate polarity-equivalence stress test.

### P21. Negation Triplet Extraction with Syntactic Dependency and Semantic Consistency (2024)

**Authors:** Yuchen Shi et al.. **Source:** [publication/preprint](https://aclanthology.org/2024.lrec-main.1058/), [source PDF](https://aclanthology.org/2024.lrec-main.1058.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Learns joint negation subject, cue and scope extraction using syntactic and semantic training.
- **Overlap / novelty consequence:** Related to reliable edit construction, not an embedding flip ranking.
- **Norm-K action:** Reference for future clause/slot-sensitive edit scope checks; do not build new extractor before confirming value.

## C. Alternative systems, confounds, structured facts, and geometry

### P22. Annotation Artifacts in Natural Language Inference Data (2018)

**Authors:** Gururangan et al.. **Source:** [publication/preprint](https://aclanthology.org/N18-2017/), [source PDF](https://aclanthology.org/N18-2017.pdf). **Evidence depth:** Primary abstract and examples inspected.

- **Method:** Hypothesis-only classifier extracts label cues in SNLI and MultiNLI, including negation and vagueness.
- **Overlap / novelty consequence:** Major threat to v4 wrong-partner AP ~0.44–0.45: the single sentence can carry the target signal even without a coherent partner.
- **Norm-K action:** Require hypothesis-only, shuffled partner, swapped-side, and edit-magnitude-only baselines on fresh sealed data.

### P23. Semantic Sensitivities and Inconsistent Predictions: Measuring the Fragility of NLI Models (2024)

**Authors:** Arakelyan, Liu & Augenstein. **Source:** [publication/preprint](https://aclanthology.org/2024.eacl-long.27/), [source PDF](https://aclanthology.org/2024.eacl-long.27.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Shows NLI prediction instability under semantically preserving surface edits, including out-of-domain tests.
- **Overlap / novelty consequence:** Conspicuous placebo risk: changing phrasing can move scores for reasons unrelated to contradiction.
- **Norm-K action:** Balance displacement with semantically equivalent rewrites and audit edit validity. Correct earlier mistaken citation to 2024.findings-emnlp.436 (that is ALIGN-SIM).

### P24. Contradiction Retrieval via Contrastive Learning with Sparsity (SparseCL) (2025)

**Authors:** Xu et al.. **Source:** [publication/preprint](https://proceedings.mlr.press/v267/xu25s.html), [source PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/xu25s/xu25s.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Trains contradiction-aware sentence embeddings and combines cosine with sparsity features for efficient retrieval.
- **Overlap / novelty consequence:** Contradiction-specific geometric metrics and sparsity are established; its encoder is specialized rather than off-the-shelf frozen.
- **Norm-K action:** A stronger alternative than adding arbitrarily many hand-built K features; compare training requirements.

### P25. eTracer: Polarity-Aware Evidence Grounding (2026)

**Authors:** Chu et al.. **Source:** [publication/preprint](https://aclanthology.org/2026.emnlp-main.314/), [source PDF](https://aclanthology.org/2026.emnlp-main.314.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Context-aware signed evidence-support/contradict/irrelevant scorer and human-labeled evidence dataset across seven domains.
- **Overlap / novelty consequence:** Closest practical alternative for evidence-based contradiction pipelines; trained and context-dependent.
- **Norm-K action:** Compare human-labeled eTracer-Bench and Menhir-style evidence accuracy/latency before considering integration.

### P26. Computation Over Geometry: Meaning Identity Is Computed, Not Shipped in the Embeddings (2026)

**Authors:** Deng. **Source:** [publication/preprint](https://arxiv.org/abs/2609.28290), [source PDF](https://arxiv.org/pdf/2609.28290). **Evidence depth:** Primary preprint abstract checked.

- **Method:** Reports that meaning identity on overlap-matched pairs often needs joint encoding; separately frozen representations and linear readers can fail.
- **Overlap / novelty consequence:** Strong theoretical and empirical threat to generalizing fixed independent vector computations, although identity is not contradiction.
- **Norm-K action:** Add paraphrase/word-order OOD tests; if performance collapses, favor joint NLI.

### P27. Semantically Rich, Metrically Affect-Blind: A Geometric Blind Spot in Modern Text Embedders (2026)

**Authors:** Moghadasi. **Source:** [publication/preprint](https://arxiv.org/abs/2610.32122), [source PDF](https://arxiv.org/pdf/2610.32122). **Evidence depth:** Primary preprint abstract checked.

- **Method:** Tests frozen encoder sentiment polarity: probe decodability can coexist with weak cosine polarity discrimination; trained corrective heads improve retrieval but hurt other metrics.
- **Overlap / novelty consequence:** Supports the recoverable-information vs visible-metric distinction; studies sentiment, not contradiction or untrained flips.
- **Norm-K action:** Contrast fixed score with learned linear probe and preserve retrieval/STS trade-offs.

### P28. Typed Relation Path Geometry in Sentence Embedding Spaces (2026)

**Authors:** Liu & Xing. **Source:** [publication/preprint](https://aclanthology.org/2026.emnlp-main.2530/), [source PDF](https://aclanthology.org/2026.emnlp-main.2530.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Studies directed typed lexical-relation paths as composed movements on the embedding sphere, with false-middle controls and transfer tests.
- **Overlap / novelty consequence:** Directed relation-vector composition is substantially established; not the particular negation flip on contradiction pairs.
- **Norm-K action:** Keep as geometric mechanism background; don't expand to universal relation arrows without transfer evidence.

### P29. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation (2023)

**Authors:** Min et al.. **Source:** [publication/preprint](https://aclanthology.org/2023.emnlp-main.741/), [source PDF](https://aclanthology.org/2023.emnlp-main.741.pdf). **Evidence depth:** Primary abstract and method passages checked.

- **Method:** Decomposes long outputs into atomic facts and checks each against evidence using retrieval and stronger models.
- **Overlap / novelty consequence:** Atomic-fact decomposition for factuality is well established; does not support a novel embedding-only contradiction detector.
- **Norm-K action:** Use as a decomposition baseline before pursuing chunk-level flip, but keep evidence/time logic explicit.

### P30. AlignScore: Evaluating Factual Consistency with A Unified Alignment Function (2023)

**Authors:** Zha et al.. **Source:** [publication/preprint](https://aclanthology.org/2023.acl-long.634/), [source PDF](https://aclanthology.org/2023.acl-long.634.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Trains an alignment model from 4.7M examples spanning diverse factuality, NLI and similarity tasks to score text consistency.
- **Overlap / novelty consequence:** Strong supervised factual alignment baseline; not training-free frozen four-vector geometry.
- **Norm-K action:** Use against operational quality/cost claims.

### P31. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (2020)

**Authors:** Khattab & Zaharia. **Source:** [publication/preprint](https://arxiv.org/abs/2004.12832), [source PDF](https://arxiv.org/pdf/2004.12832). **Evidence depth:** Primary bibliographic link; architecture broadly established.

- **Method:** Scores retrieval with contextual token embeddings and late-interaction matching instead of one sentence vector.
- **Overlap / novelty consequence:** Fine-grained token-level matching is established; retrieval != contradiction.
- **Norm-K action:** Only pursue token/chunk/slot comparisons if whole-sentence flip passes out-of-domain checks.

### P32. Investigating Semantic Subspaces of Transformer Sentence Embeddings through Linear Structural Probing (2023)

**Authors:** Nikolaev & Padó. **Source:** [publication/preprint](https://aclanthology.org/2023.blackboxnlp-1.11/), [source PDF](https://aclanthology.org/2023.blackboxnlp-1.11.pdf). **Evidence depth:** Primary abstract checked.

- **Method:** Learns linear structural-probe subspaces for sentence semantic similarity and NLI.
- **Overlap / novelty consequence:** Learned semantic subspaces and NLI probing are not novel; differs from a frozen parameter-free flip.
- **Norm-K action:** Park additional low-rank/sparsity/asinh features until causal flip controls pass.

### P33. STALE: Can LLM Agents Know When Their Memories Are No Longer Valid? (2026)

**Authors:** Chao et al.. **Source:** [publication/preprint](https://arxiv.org/abs/2605.06527), [source PDF](https://arxiv.org/pdf/2605.06527). **Evidence depth:** Primary preprint abstract checked.

- **Method:** Benchmarks implicit time-dependent memory invalidation and stale-state handling rather than pure textual opposition.
- **Overlap / novelty consequence:** Shows why temporal conflict resolution needs structured belief state beyond cosine-based contradiction ranking.
- **Norm-K action:** Keep Menhir temporal-belief lane independent of success/failure of flip.

## Research hygiene and corrections

- The 2024 **Semantic Sensitivities** paper is at [2024.eacl-long.27](https://aclanthology.org/2024.eacl-long.27/), **not** 2024.findings-emnlp.436 (which is ALIGN-SIM). Previous responses conflated them.
- The February 2026 Scientific Reports paper's title is **Computation of sentence similarity score through hybrid deep learning with a special focus on negation sentence**; "NAS Scorer" is its method name, not the article title.
- Cui 2020 §2.2 and §4.2 confirm plain two-vector cosine classification by thresholds after multimodal contrastive pretraining. It does **not** do Norm-K-style negation edits at inference; the one-paragraph description in Schumann–Marx Gómez was accurate but incomplete.
- For the OpenReview entailment-operator submission, final publication metadata and peer-review disposition must be confirmed. For arXiv preprints and recent workshop papers, do not assume peer-reviewed novelty decisions.
- No paper-level AP/accuracy numbers are directly comparable across differing datasets, prevalence, labels, and metrics; this audit does not provide a meta-analysis.
- Source files are cited by original publisher links. If local PDF mirroring is desired, first inspect each item's redistribution license and add checksums and citation metadata. **No licenses are assumed.**

**Bottom line:** Keep the current flip result as a narrow, auditable lead for one falsification-oriented round. We have research space for identifying and testing the cause of the response, but not a justified basis for an open-ended standalone contradiction product or for Menhir integration.
