# Reference Code Library

This folder is a **deduplicated, runnable code library** auto-extracted from
`AI_Learning_Cursor` (the 33,511-line transcript of the original Cursor
session that produced this curriculum).


| Stat                               | Value                                   |
| ---------------------------------- | --------------------------------------- |
| Original code blocks in transcript | 202                                     |
| Duplicates removed                 | 101                                     |
| Trivial summary blocks skipped     | 21                                      |
| **Unique files written**           | **80**                                  |
| Python files that compile cleanly  | **77 / 77 (100%)**                      |
| Folders                            | 9 (one per phase of the new curriculum) |


---

## How to use this folder

> **Rule #1: Build it yourself FIRST. Diff against this code SECOND.**

Every project in the project ladder maps to one or more reference files here.
The recommended workflow for each project is:

1. **Read the corresponding phase's `.md` file** — that's where the *theory* lives.
2. **Attempt the project from scratch in your `zero-to-ai-engineer/` repo** — this is the only way mastery actually transfers (see `LEARNING_SYSTEM.md` § Four-Gate Test).
3. **Run `diff your_solution.py reference_code/<phase>/<file>.py`** — find what you missed, what you over-engineered, and what the reference does more cleanly.
4. **Commit your version + a `NOTES.md` explaining the diff** — that's how you internalize the lesson.

If you read the reference code first, you will trick yourself into thinking
you understand things you only *recognize*. Resist.

---

## Setup notes (do this once)

Every reference file assumes you are running inside the conda env from
`Phase_0_Environment_and_Mindset_Setup.md`:

```bash
conda activate ai-engineer            # or whatever you named it
pip install torch torchvision matplotlib numpy scikit-learn pandas \
            transformers datasets accelerate peft trl \
            sentence-transformers chromadb fastapi uvicorn pydantic \
            jupyter notebook tiktoken xgboost
```

For Apple Silicon (M3 Pro), PyTorch should pick up MPS automatically — every
file that needs the GPU does:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## Folder ↔ Curriculum map


| Folder                         | New-curriculum phase               | Weeks       | When to read                      |
| ------------------------------ | ---------------------------------- | ----------- | --------------------------------- |
| `phase_0_environment/`         | Phase 0: Environment & Mindset     | Week 0      | Before starting Week 1            |
| `phase_1_math_engine/`         | Phase 1: The Math Engine           | Weeks 1–6   | After each weekly assignment      |
| `phase_2_classical_ml/`        | Phase 2: Classical ML              | Weeks 7–9   | After your own implementation     |
| `phase_3_neural_networks/`     | Phase 3: Neural Networks Deep Dive | Weeks 10–15 | After your own implementation     |
| `phase_4_nlp_and_sequences/`   | Phase 4: NLP & Sequence Models     | Weeks 16–20 | After your own implementation     |
| `phase_5_transformer/`         | Phase 5: The Transformer           | Weeks 21–25 | After your own implementation     |
| `phase_6_mini_gpt/`            | Phase 6: Build Your Mini-GPT       | Weeks 26–29 | After your own implementation     |
| `phase_7_ai_engineer_toolkit/` | Phase 7: AI Engineer Toolkit       | Weeks 30–34 | Reference + scaffold for plumbing |
| `phase_8_capstone/`            | Phase 8: Capstone & Launch         | Weeks 35–36 | Inspiration + interview prep      |


---

## File ↔ Curriculum index

Project IDs (P1–P18) refer to the project ladder in
`01_ZERO_TO_GPT_6_Month_Masterclass.md` § "THE PROJECT LADDER".

`When` legend: **A** = read AFTER your attempt (default). **B** = read BEFORE
(only when the file is plumbing/boilerplate that doesn't teach by re-deriving).

---

### `phase_0_environment/` — Week 0


| File                              | What it is                                                   | Curriculum slot     | When |
| --------------------------------- | ------------------------------------------------------------ | ------------------- | ---- |
| `01_mac_m3_pro_setup_commands.sh` | brew + conda env + pytorch install commands                  | Phase 0 §1 (Setup)  | B    |
| `02_pytorch_mps_check.py`         | Verifies MPS is available; prints device + tensor smoke test | Phase 0 §2 (Verify) | B    |


---

### `phase_1_math_engine/` — Weeks 1–6


| File                                         | What it is                                          | Curriculum slot     | When |
| -------------------------------------------- | --------------------------------------------------- | ------------------- | ---- |
| `01_numpy_salesforce_analogy.py`             | Salesforce data → NumPy array intro                 | Week 1 Day 1        | A    |
| `02_creating_arrays_and_init.py`             | `np.zeros` / `np.ones` / random / init patterns     | Week 1 Day 1        | B    |
| `03_dot_product_drills.py`                   | Vector dot product as "neuron firing"               | Week 1 Day 2        | A    |
| `04_matrix_multiplication.py`                | `(N, F) @ (F, H)` = batched neuron layer            | Week 1 Day 3        | A    |
| `05_broadcasting.py`                         | Adding biases / per-feature scaling                 | Week 1 Day 4        | A    |
| `05a_reshape_patterns.py`                    | Image (H, W) → flat (H·W) → batch (B, H·W)          | Week 1 Day 4        | A    |
| `06_matrix_operations_library_assignment.py` | **Project #1**: build a matrix-ops lib from scratch | Week 1 (Project P1) | A    |
| `07_python_pattern_class_neuron.py`          | OO `Neuron` class — boilerplate you'll reuse        | Week 1 Day 5        | B    |
| `07a_python_pattern_list_comprehensions.py`  | Vectorized data preprocessing                       | Week 1 Day 6        | A    |
| `07b_python_pattern_batch_generator.py`      | `yield` for memory-efficient batching               | Week 1 Day 6        | A    |
| `07c_python_pattern_context_manager.py`      | `with` blocks for gradient tracking                 | Week 1 Day 7        | A    |
| `08_vectors_intro.py`                        | Geometric meaning of vectors (point vs direction)   | Week 2 Day 1        | A    |
| `09_vector_operations.py`                    | Norm, addition, scaling, projection                 | Week 2 Day 2        | A    |
| `10_visualizing_vectors.py`                  | matplotlib quiver plots of vector ops               | Week 2 Day 2        | A    |
| `11_customer_similarity_engine.py`           | **Project #1 alt**: cosine-similarity engine        | Week 2 (Project P1) | A    |
| `12_matrices_as_transformations.py`          | 2×2 matrices as scaling/rotation/shear              | Week 2 Day 3        | A    |
| `13_layer_as_matmul.py`                      | `output = activation(input @ W + b)` derived        | Week 2 Day 4        | A    |
| `14_matrix_properties.py`                    | Transpose, inverse, determinant, eigenvalues        | Week 2 Day 5        | A    |
| `15_matrix_transformation_visualizer.py`     | Interactive viz of matrix transforms                | Week 2 (Project P1) | A    |
| `16_derivatives_intro.py`                    | f'(x) numerically + symbolically                    | Week 3 Day 1        | A    |
| `18_gradient_descent_from_scratch.py`        | **Project #2**: 1D gradient descent loop            | Week 3 (Project P2) | A    |
| `18a_gradient_descent_2d_assignment.py`      | 2D extension — find minimum of a surface            | Week 3 (Project P2) | A    |
| `19_chain_rule.py`                           | The chain rule = backprop foundation                | Week 4 Day 1        | A    |


> **Phase 1 gap-fill:** the new curriculum's Weeks 5–6 (Probability + Information Theory) are NOT in the original transcript — they're additions in the rewritten `Phase_1_The_Math_Engine.md`. There is intentionally no reference code here for those weeks because the curriculum's `.md` already contains the from-scratch implementations of `bernoulli_pmf`, `gaussian_pdf`, `cross_entropy`, `softmax`, etc.

---

### `phase_2_classical_ml/` — Weeks 7–9


| File                                     | What it is                                                   | Curriculum slot     | When |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------- | ---- |
| `01_ml_framework_overview.py`            | The "data → model → loss → optimizer" loop                   | Week 7 Day 1        | A    |
| `02_types_of_ml.py`                      | Supervised / unsupervised / RL — examples of each            | Week 7 Day 1        | A    |
| `03_linear_regression_from_scratch.py`   | **Project #4 part 1**: linreg in NumPy, validated vs sklearn | Week 7 (Project P4) | A    |
| `04_logistic_regression_from_scratch.py` | **Project #4 part 2**: logreg in NumPy, BCE loss             | Week 7 (Project P4) | A    |


> **Phase 2 gap-fill:** Weeks 8–9 (Trees / Boosting / Metrics / Cross-Validation / Spam-Classifier ladder) are NOT in the original transcript — they're additions in the rewritten `Phase_2_Classical_ML.md`. Implement those from scratch using the `.md` as your guide.

---

### `phase_3_neural_networks/` — Weeks 10–15


| File                                        | What it is                                         | Curriculum slot       | When |
| ------------------------------------------- | -------------------------------------------------- | --------------------- | ---- |
| `01_single_neuron_from_scratch.py`          | Forward + backward pass of one neuron              | Week 10 Day 1         | A    |
| `02_mlp_from_scratch.py`                    | **Project #3** & **Project #6**: full MLP in NumPy | Week 11 (Project P3)  | A    |
| `03_pytorch_tensors_and_mps.py`             | PyTorch tensors + MPS GPU acceleration             | Week 12 Day 1         | A    |
| `04_nn_module_pattern.py`                   | The canonical `nn.Module` subclass pattern         | Week 12 Day 2         | A    |
| `05_circle_classification.py`               | End-to-end PyTorch classifier on toy 2D data       | Week 12 Day 3         | A    |
| `06_mnist_data_loading.py`                  | `torchvision` MNIST loaders + visualization        | Week 12 Day 4         | A    |
| `07_mnist_classifier.py`                    | **Project #6**: MNIST MLP, full training loop      | Week 12 (Project P6)  | A    |
| `07a_mnist_classifier_with_augmentation.py` | Same task + transforms + dropout                   | Week 13 (Project P6)  | A    |
| `08_data_augmentation.py`                   | `torchvision.transforms` pipeline                  | Week 13 Day 2         | A    |
| `09_understanding_convolutions.py`          | Conv-as-feature-detector visualizations            | Week 14 Day 1         | A    |
| `11_batchnorm_dropout_optimizers.py`        | BN + dropout + SGD/Momentum/Adam compared          | Week 13 (Project P6+) | A    |


> **Phase 3 gap-fill:** the new curriculum has a "Training Diagnostics / Loss-curve Pathology Zoo" mini-week (Week 13.5) that's NOT in the transcript. Build that yourself from the `.md`.

---

### `phase_4_nlp_and_sequences/` — Weeks 16–20


| File                                | What it is                                                 | Curriculum slot       | When |
| ----------------------------------- | ---------------------------------------------------------- | --------------------- | ---- |
| `02_text_preprocessing_basics.py`   | Tokenization, vocabulary building, encoding                | Week 16 Day 1         | A    |
| `03_word2vec_skipgram.py`           | **Project #9**: Skip-gram + negative sampling, t-SNE viz   | Week 16 (Project P9)  | A    |
| `04_glove_pretrained_embeddings.py` | Loading + querying pre-trained GloVe vectors               | Week 17 Day 1         | A    |
| `05_rnn_from_scratch.py`            | Vanilla RNN: forward, BPTT by hand                         | Week 18 Day 1         | A    |
| `06_char_rnn_text_generator.py`     | **Project #10**: char-level RNN that generates text        | Week 18 (Project P10) | A    |
| `07_vanishing_gradient_problem.py`  | Empirical demo of why RNNs struggle                        | Week 18 Day 4         | A    |
| `08_lstm_from_scratch.py`           | LSTM cell built gate-by-gate (no `nn.LSTM`)                | Week 19 Day 1         | A    |
| `09_lstm_sentiment_classifier.py`   | LSTM + classifier head on IMDb-style data                  | Week 19 Day 3         | A    |
| `10_seq2seq_translation.py`         | Encoder-decoder for tiny English→reverse task              | Week 20 Day 1         | A    |
| `11_attention_preview_bahdanau.py`  | Bahdanau attention bolted onto seq2seq — bridge to Phase 5 | Week 20 Day 4         | A    |


---

### `phase_5_transformer/` — Weeks 21–25


| File                                     | What it is                                       | Curriculum slot       | When |
| ---------------------------------------- | ------------------------------------------------ | --------------------- | ---- |
| `01_self_attention_from_scratch.py`      | **Project #12**: Q/K/V attention in pure PyTorch | Week 21 (Project P12) | A    |
| `02_attention_mechanics_deep_dive.py`    | Attention with weight visualization              | Week 21 Day 4         | A    |
| `03_production_ready_self_attention.py`  | Same, but with masking, scaling, dropout         | Week 21 Day 5         | A    |
| `04_multi_head_attention.py`             | **Project #13**: split heads, concat, project    | Week 22 (Project P13) | A    |
| `05_positional_encoding.py`              | Sinusoidal positional encoding + plot            | Week 22 Day 3         | A    |
| `06_complete_attention_layer.py`         | MHA + PE + residual + LayerNorm in one block     | Week 23 Day 1         | A    |
| `08_transformer_encoder.py`              | Stack of encoder blocks with FFN                 | Week 23 Day 2         | A    |
| `09_transformer_decoder.py`              | Decoder with causal mask + cross-attention       | Week 24 Day 1         | A    |
| `10_complete_transformer_translation.py` | **Project #14**: full encoder-decoder translator | Week 24 (Project P14) | A    |
| `11_train_transformer_pipeline.py`       | Training loop: warmup, label smoothing, eval     | Week 24 Day 5         | B    |


> **Phase 5 gap-fill:** Week 25 (Mechanistic Interpretability with TransformerLens) is in the new curriculum but not in the transcript — there's no reference code for it.

---

### `phase_6_mini_gpt/` — Weeks 26–29


| File                                  | What it is                                              | Curriculum slot       | When |
| ------------------------------------- | ------------------------------------------------------- | --------------------- | ---- |
| `01_gpt_architecture_from_scratch.py` | **Project #15**: decoder-only GPT, configurable size    | Week 26 (Project P15) | A    |
| `02_language_modeling_objective.py`   | Next-token prediction loss, label shifting              | Week 26 Day 3         | A    |
| `03_gpt_training_infrastructure.py`   | Optimizer, scheduler, checkpointing, logging            | Week 27 Day 1         | B    |
| `04_bpe_tokenizer_from_scratch.py`    | **Project #11**: full BPE training + encode/decode      | Week 27 (Project P11) | A    |
| `05_dataset_preparation_for_gpt.py`   | Streaming dataset, packing, batch sampling              | Week 27 Day 4         | B    |
| `06_train_gpt_on_shakespeare.py`      | **Project #15 main**: end-to-end training run on M3 Pro | Week 28 (Project P15) | A    |
| `07_text_generation_and_sampling.py`  | Greedy / temperature / top-k / top-p / nucleus          | Week 29 Day 1         | A    |
| `08_complete_minigpt_package.py`      | Everything packaged as installable module               | Week 29 Day 5         | B    |


---

### `phase_7_ai_engineer_toolkit/` — Weeks 30–34


| File                              | What it is                                                | Curriculum slot       | When |
| --------------------------------- | --------------------------------------------------------- | --------------------- | ---- |
| `01_finetune_gpt2_huggingface.py` | **Project #16**: SFT GPT-2 on custom corpus via `Trainer` | Week 30 (Project P16) | B    |
| `02_lora_efficient_finetuning.py` | **Project #16 alt**: LoRA via `peft` library              | Week 30 (Project P16) | B    |
| `03_rlhf_three_stages.py`         | Conceptual SFT → Reward Model → PPO walkthrough           | Week 34 Day 3         | A    |
| `04_rlhf_end_to_end_example.py`   | Minimal end-to-end RLHF on tiny model                     | Week 34 Day 4         | A    |
| `07_fastapi_serving.py`           | **Project #17**: model-serving REST API                   | Week 34 (Project P17) | B    |
| `07a_deployment_commands.py`      | curl examples + uvicorn launch commands                   | Week 34 (Project P17) | B    |
| `08_docker_deployment.Dockerfile` | Production Dockerfile for the inference service           | Week 34 (Project P17) | B    |
| `08a_docker_compose.yaml`         | docker-compose.yaml: API + Redis (cache)                  | Week 34 (Project P17) | B    |
| `09_production_best_practices.py` | Logging, error handling, request validation               | Week 34 Day 5         | B    |


> **Phase 7 gap-fill:** the new curriculum's RAG / Vector DB / Agents / Evals / Observability content (Weeks 31–33) is NOT in the original transcript — those are additions in `Phase_7_AI_Engineer_Toolkit.md`. The `.md` already contains complete reference implementations using Chroma, sentence-transformers, BM25, RRF, CrossEncoder reranker, Ragas, ReAct agents, Pydantic+Instructor, Langfuse, etc.

---

### `phase_8_capstone/` — Weeks 35–36


| File                         | What it is                                       | Curriculum slot | When |
| ---------------------------- | ------------------------------------------------ | --------------- | ---- |
| `01_portfolio_template.py`   | Template structure for portfolio README + blog   | Week 36 Day 1   | B    |
| `02_interview_prep_guide.py` | Common AI engineer interview questions + answers | Week 36 Day 3   | A    |
| `03_career_roadmap.py`       | 0–6 / 6–12 / 12–24 month career milestones       | Week 36 Day 5   | A    |


---

## What's NOT in this folder (and why)

The new 9-phase curriculum is a strict superset of the original 6-phase
transcript. The following topics exist in the new `.md` files but have **no
reference code here** because the original Cursor session never produced code
for them. You'll write your own from scratch using the `.md` as the guide:


| Phase   | Topic                                                       | Reference file location instead                        |
| ------- | ----------------------------------------------------------- | ------------------------------------------------------ |
| Phase 1 | Probability + Information Theory (Weeks 5–6)                | `Phase_1_The_Math_Engine.md` itself contains full code |
| Phase 2 | Trees / Boosting / Metrics / CV / Spam Ladder (Weeks 8–9)   | `Phase_2_Classical_ML.md` itself contains full code    |
| Phase 3 | Loss-curve Pathology Zoo (Week 13.5)                        | `Phase_3_Neural_Networks.md` itself                    |
| Phase 5 | Mechanistic Interpretability with TransformerLens (Week 25) | `Phase_5_The_Transformer.md` itself                    |
| Phase 6 | Scaling Laws sizing exercise (pre-Week 26)                  | `Phase_6_Build_Your_GPT.md` itself                     |
| Phase 7 | RAG / Vector DB / Hybrid Search / Reranking (Week 31)       | `Phase_7_AI_Engineer_Toolkit.md` itself                |
| Phase 7 | Agents / Tools / Pydantic / ReAct (Week 32)                 | `Phase_7_AI_Engineer_Toolkit.md` itself                |
| Phase 7 | Evals / LLM-as-judge / PromptFoo / Langfuse (Week 33)       | `Phase_7_AI_Engineer_Toolkit.md` itself                |
| Phase 7 | Quantization / `llama.cpp` / Q4_K_M (Week 34)               | `Phase_7_AI_Engineer_Toolkit.md` itself                |
| Phase 8 | `CodeSage` capstone integration                             | `Phase_8_Capstone_and_Launch.md` itself                |


---

## How to regenerate this folder

If you ever change `AI_Learning_Cursor` or want to re-extract:

```bash
# 1) Wipe current output
rm -rf reference_code/phase_* reference_code/_unsorted reference_code/_extraction_report.txt

# 2) Re-extract
python3 reference_code/_extract_from_transcript.py

# 3) Apply the human-curated rename pass for collisions
python3 reference_code/_finalize_filenames.py

# 4) Verify everything still parses
python3 -c "
import py_compile, pathlib
for p in pathlib.Path('reference_code').rglob('*.py'):
    if not p.name.startswith('_'):
        py_compile.compile(str(p), doraise=True)
print('OK')
"
```

The audit log of what was extracted lands in `_extraction_report.txt`.

---

## Provenance

Every `.py` file in this folder begins with a comment header like:

```python
# Source: AI_Learning_Cursor lines 22810-23252
# Original transcript phase: 5 - BUILD YOUR OWN GPT
# Nearest header: #### CODE: GPT Architecture
# Title: GPT ARCHITECTURE FROM SCRATCH
```

so you can always trace any file back to its exact location in the transcript
and re-read the surrounding chat context if you want more explanation.