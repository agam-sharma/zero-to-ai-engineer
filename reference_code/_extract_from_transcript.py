"""
Extract all unique code blocks from AI_Learning_Cursor (the original Cursor
chat transcript) into a clean, deduplicated, runnable reference library
organized by the NEW 9-phase curriculum.

Run:
    python3 reference_code/_extract_from_transcript.py

Output:
    reference_code/<phase_folder>/<NN_descriptive_name>.<ext>
    reference_code/README.md
    reference_code/_extraction_report.txt   (raw audit of what was found)
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = REPO_ROOT / "AI_Learning_Cursor"
OUT_ROOT = REPO_ROOT / "reference_code"
REPORT = OUT_ROOT / "_extraction_report.txt"
README = OUT_ROOT / "README.md"

# Map LANGUAGE -> file extension
LANG_EXT = {
    "python": "py",
    "py": "py",
    "bash": "sh",
    "shellscript": "sh",
    "shell": "sh",
    "yaml": "yaml",
    "yml": "yaml",
    "dockerfile": "Dockerfile",
    "docker": "Dockerfile",
    "plain": "txt",
    "plaintext": "txt",
    "txt": "txt",
}

# Transcript "phase" headers we want to detect.
# These come from the original (6-phase) transcript.
PHASE_HEADER_RE = re.compile(r"^#\s+PHASE\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
WEEK_HEADER_RE = re.compile(r"^##\s+WEEK\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
GENERIC_H_RE = re.compile(r"^(#{1,4})\s+(.+)$")

# ---------------------------------------------------------------------------
# Mapping: TOPIC SLUG  ->  (new_phase_folder, ordering_index, friendly_filename, project_id_or_None)
#
# The TOPIC SLUG is derived from each block's leading docstring title
# (or from the nearest preceding markdown header if no docstring is found).
#
# Anything not in this table will land in `reference_code/_unsorted/` and
# show up in the extraction report so we can re-classify it.
# ---------------------------------------------------------------------------

# Helper for the table below
def _t(slug, folder, order, fname, project=None):
    return (slug.lower(), folder, order, fname, project)


# NOTE: Order matters. The classifier scans this dict in INSERTION order and
# uses the FIRST slug that is a substring of the docstring title (or, failing
# that, the nearest header). So put MORE SPECIFIC slugs BEFORE more generic ones.
TOPIC_MAP: dict[str, tuple[str, int, str, str | None]] = {}
for slug, folder, order, fname, project in [
    # ============================================================
    # Phase 0: Environment & Setup
    # ============================================================
    _t("step-by-step setup", "phase_0_environment", 1, "01_mac_m3_pro_setup_commands", None),
    _t("setup guide", "phase_0_environment", 1, "01_mac_m3_pro_setup_commands", None),
    _t("environment setup", "phase_0_environment", 1, "01_mac_m3_pro_setup_commands", None),
    _t("conda environment", "phase_0_environment", 2, "02_conda_env_create", None),
    _t("pytorch mps setup", "phase_0_environment", 3, "03_pytorch_mps_check", None),
    _t("verify pytorch", "phase_0_environment", 3, "03_pytorch_mps_check", None),
    _t("environment ready", "phase_0_environment", 4, "04_env_smoketest", None),
    # heuristic: very-early Python blocks under setup-guide header
    _t("apple silicon", "phase_0_environment", 5, "05_apple_silicon_notes", None),

    # ============================================================
    # Phase 1: Math Engine
    # ============================================================
    # NumPy core (lines ~370-504 in transcript)
    _t("salesforce analogy", "phase_1_math_engine", 1, "01_numpy_salesforce_analogy", None),
    _t("core numpy concepts", "phase_1_math_engine", 2, "02_core_numpy_concepts", None),
    _t("numpy mastery for ai", "phase_1_math_engine", 2, "02_core_numpy_concepts", None),
    _t("dot product", "phase_1_math_engine", 3, "03_dot_product_drills", None),
    _t("matrix multiplication", "phase_1_math_engine", 4, "04_matrix_multiplication", None),
    _t("broadcasting", "phase_1_math_engine", 5, "05_broadcasting", None),
    _t("matrix operations library", "phase_1_math_engine", 6, "06_matrix_operations_library_assignment", "P1"),
    # Python patterns (~line 642+)
    _t("python patterns used in ai", "phase_1_math_engine", 7, "07_python_patterns_for_ai", None),
    _t("python patterns for ai", "phase_1_math_engine", 7, "07_python_patterns_for_ai", None),
    _t("python refresher", "phase_1_math_engine", 7, "07_python_patterns_for_ai", None),
    # Vectors (~line 808+)
    _t("theory: what is a vector", "phase_1_math_engine", 8, "08_vectors_intro", None),
    _t("what is a vector", "phase_1_math_engine", 8, "08_vectors_intro", None),
    _t("key vector operations", "phase_1_math_engine", 9, "09_vector_operations", None),
    _t("vectors building blocks", "phase_1_math_engine", 9, "09_vector_operations", None),
    _t("visualizing vectors", "phase_1_math_engine", 10, "10_visualizing_vectors", None),
    _t("customer similarity", "phase_1_math_engine", 11, "11_customer_similarity_engine", "P1"),
    # Matrices (~line 1060+)
    _t("matrices transformations", "phase_1_math_engine", 12, "12_matrices_as_transformations", None),
    _t("neural network connection", "phase_1_math_engine", 13, "13_layer_as_matmul", None),
    _t("matrix properties", "phase_1_math_engine", 14, "14_matrix_properties", None),
    _t("transformation visualizer", "phase_1_math_engine", 15, "15_matrix_transformation_visualizer", None),
    # Calculus (~line 1319+)
    _t("derivatives: the direction", "phase_1_math_engine", 16, "16_derivatives_intro", None),
    _t("derivatives", "phase_1_math_engine", 16, "16_derivatives_intro", None),
    _t("partial derivatives", "phase_1_math_engine", 17, "17_partial_derivatives", None),
    _t("gradient descent", "phase_1_math_engine", 18, "18_gradient_descent_from_scratch", "P2"),
    _t("chain rule", "phase_1_math_engine", 19, "19_chain_rule", None),
    _t("backpropagation", "phase_1_math_engine", 20, "20_backprop_by_hand", None),
    _t("micrograd", "phase_1_math_engine", 21, "21_micrograd_walkthrough", None),
    _t("autograd", "phase_1_math_engine", 22, "22_autograd_intro", None),
    _t("numpy neural network", "phase_1_math_engine", 23, "23_numpy_neural_network_simulator", "P3"),

    # ============================================================
    # Phase 2: Classical ML  (NEW phase; transcript covered linreg/logreg under "Phase 1")
    # ============================================================
    _t("machine learning framework", "phase_2_classical_ml", 1, "01_ml_framework_overview", None),
    _t("ml framework", "phase_2_classical_ml", 1, "01_ml_framework_overview", None),
    _t("types of machine learning", "phase_2_classical_ml", 2, "02_types_of_ml", None),
    _t("linear regression from scratch", "phase_2_classical_ml", 3, "03_linear_regression_from_scratch", "P4"),
    _t("linear regression", "phase_2_classical_ml", 3, "03_linear_regression_from_scratch", "P4"),
    _t("logistic regression", "phase_2_classical_ml", 4, "04_logistic_regression_from_scratch", "P4"),

    # ============================================================
    # Phase 3: Neural Networks (PyTorch + CNNs)
    # ============================================================
    # From-scratch NN (transcript Phase 1 territory)
    _t("the neuron", "phase_3_neural_networks", 1, "01_single_neuron_from_scratch", None),
    _t("multi-layer neural network", "phase_3_neural_networks", 2, "02_mlp_from_scratch", "P3"),
    _t("multi-layer neural network from scratch", "phase_3_neural_networks", 2, "02_mlp_from_scratch", "P3"),
    # PyTorch
    _t("pytorch fundamentals", "phase_3_neural_networks", 3, "03_pytorch_tensors_and_mps", None),
    _t("pytorch basics and mps", "phase_3_neural_networks", 3, "03_pytorch_tensors_and_mps", None),
    _t("pytorch tensors", "phase_3_neural_networks", 3, "03_pytorch_tensors_and_mps", None),
    _t("neural networks with nn.module", "phase_3_neural_networks", 4, "04_nn_module_pattern", None),
    _t("nn.module", "phase_3_neural_networks", 4, "04_nn_module_pattern", None),
    _t("complete classifier example", "phase_3_neural_networks", 5, "05_circle_classification", None),
    _t("circle classification", "phase_3_neural_networks", 5, "05_circle_classification", None),
    # MNIST + CNNs
    _t("mnist dataset loading", "phase_3_neural_networks", 6, "06_mnist_data_loading", "P6"),
    _t("mnist digit classifier", "phase_3_neural_networks", 7, "07_mnist_classifier", "P6"),
    _t("mnist", "phase_3_neural_networks", 7, "07_mnist_classifier", "P6"),
    _t("data augmentation and regularization", "phase_3_neural_networks", 8, "08_data_augmentation", None),
    _t("data augmentation", "phase_3_neural_networks", 8, "08_data_augmentation", None),
    _t("understanding convolutions", "phase_3_neural_networks", 9, "09_understanding_convolutions", None),
    _t("convolutional neural networks", "phase_3_neural_networks", 10, "10_cnn_image_classifier", "P7"),
    _t("cnn for mnist", "phase_3_neural_networks", 10, "10_cnn_image_classifier", "P7"),
    _t("cnn implementation", "phase_3_neural_networks", 10, "10_cnn_image_classifier", "P7"),
    # Modern training
    _t("modern training techniques", "phase_3_neural_networks", 11, "11_batchnorm_dropout_optimizers", None),
    _t("optimizers", "phase_3_neural_networks", 12, "12_optimizers_sgd_momentum_adam", None),
    _t("dropout", "phase_3_neural_networks", 13, "13_dropout_regularization", None),
    _t("batch normalization", "phase_3_neural_networks", 14, "14_batch_normalization", None),

    # ============================================================
    # Phase 4: NLP & Sequence Models
    # ============================================================
    _t("evolution to transformers", "phase_4_nlp_and_sequences", 1, "01_evolution_overview", None),
    _t("text preprocessing and tokenization", "phase_4_nlp_and_sequences", 2, "02_text_preprocessing_basics", None),
    _t("text preprocessing", "phase_4_nlp_and_sequences", 2, "02_text_preprocessing_basics", None),
    _t("word2vec from scratch", "phase_4_nlp_and_sequences", 3, "03_word2vec_skipgram", "P9"),
    _t("word2vec", "phase_4_nlp_and_sequences", 3, "03_word2vec_skipgram", "P9"),
    _t("using pre-trained embeddings", "phase_4_nlp_and_sequences", 4, "04_glove_pretrained_embeddings", None),
    _t("glove", "phase_4_nlp_and_sequences", 4, "04_glove_pretrained_embeddings", None),
    _t("recurrent neural networks", "phase_4_nlp_and_sequences", 5, "05_rnn_from_scratch", None),
    _t("rnn from scratch", "phase_4_nlp_and_sequences", 5, "05_rnn_from_scratch", None),
    _t("character-level text generation", "phase_4_nlp_and_sequences", 6, "06_char_rnn_text_generator", "P10"),
    _t("character-level text generator", "phase_4_nlp_and_sequences", 6, "06_char_rnn_text_generator", "P10"),
    _t("vanishing gradient", "phase_4_nlp_and_sequences", 7, "07_vanishing_gradient_problem", None),
    _t("lstm (long short-term memory)", "phase_4_nlp_and_sequences", 8, "08_lstm_from_scratch", None),
    _t("lstm from scratch", "phase_4_nlp_and_sequences", 8, "08_lstm_from_scratch", None),
    _t("long short-term memory", "phase_4_nlp_and_sequences", 8, "08_lstm_from_scratch", None),
    _t("sentiment classification with lstm", "phase_4_nlp_and_sequences", 9, "09_lstm_sentiment_classifier", None),
    _t("lstm sentiment", "phase_4_nlp_and_sequences", 9, "09_lstm_sentiment_classifier", None),
    _t("sentiment classifier", "phase_4_nlp_and_sequences", 9, "09_lstm_sentiment_classifier", None),
    _t("sequence-to-sequence", "phase_4_nlp_and_sequences", 10, "10_seq2seq_translation", None),
    _t("seq2seq", "phase_4_nlp_and_sequences", 10, "10_seq2seq_translation", None),
    _t("attention mechanism preview", "phase_4_nlp_and_sequences", 11, "11_attention_preview_bahdanau", None),
    _t("attention mechanism", "phase_4_nlp_and_sequences", 11, "11_attention_preview_bahdanau", None),
    _t("bahdanau attention", "phase_4_nlp_and_sequences", 11, "11_attention_preview_bahdanau", None),

    # ============================================================
    # Phase 5: Transformer
    # ============================================================
    _t("self-attention from scratch", "phase_5_transformer", 1, "01_self_attention_from_scratch", "P12"),
    _t("self attention", "phase_5_transformer", 1, "01_self_attention_from_scratch", "P12"),
    _t("attention mechanics deep dive", "phase_5_transformer", 2, "02_attention_mechanics_deep_dive", None),
    _t("production-ready self-attention", "phase_5_transformer", 3, "03_production_ready_self_attention", None),
    _t("multi-head attention", "phase_5_transformer", 4, "04_multi_head_attention", "P13"),
    _t("positional encoding", "phase_5_transformer", 5, "05_positional_encoding", None),
    _t("position encoding", "phase_5_transformer", 5, "05_positional_encoding", None),
    _t("complete attention layer", "phase_5_transformer", 6, "06_complete_attention_layer", None),
    _t("transformer block", "phase_5_transformer", 7, "07_transformer_block", None),
    _t("transformer encoder", "phase_5_transformer", 8, "08_transformer_encoder", None),
    _t("transformer decoder", "phase_5_transformer", 9, "09_transformer_decoder", None),
    _t("complete transformer", "phase_5_transformer", 10, "10_complete_transformer_translation", "P14"),
    _t("training a transformer", "phase_5_transformer", 11, "11_train_transformer_pipeline", None),

    # ============================================================
    # Phase 6: Mini-GPT
    # ============================================================
    _t("gpt architecture from scratch", "phase_6_mini_gpt", 1, "01_gpt_architecture_from_scratch", "P15"),
    _t("gpt architecture", "phase_6_mini_gpt", 1, "01_gpt_architecture_from_scratch", "P15"),
    _t("language modeling objective", "phase_6_mini_gpt", 2, "02_language_modeling_objective", None),
    _t("gpt training infrastructure", "phase_6_mini_gpt", 3, "03_gpt_training_infrastructure", None),
    _t("tokenization for gpt", "phase_6_mini_gpt", 4, "04_bpe_tokenizer_from_scratch", "P11"),
    _t("tokenization", "phase_6_mini_gpt", 4, "04_bpe_tokenizer_from_scratch", "P11"),
    _t("byte pair encoding", "phase_6_mini_gpt", 4, "04_bpe_tokenizer_from_scratch", "P11"),
    _t("bpe", "phase_6_mini_gpt", 4, "04_bpe_tokenizer_from_scratch", "P11"),
    _t("preparing data for gpt", "phase_6_mini_gpt", 5, "05_dataset_preparation_for_gpt", None),
    _t("dataset preparation", "phase_6_mini_gpt", 5, "05_dataset_preparation_for_gpt", None),
    _t("training your gpt on shakespeare", "phase_6_mini_gpt", 6, "06_train_gpt_on_shakespeare", "P15"),
    _t("training your gpt", "phase_6_mini_gpt", 6, "06_train_gpt_on_shakespeare", "P15"),
    _t("text generation strategies", "phase_6_mini_gpt", 7, "07_text_generation_and_sampling", None),
    _t("text generation", "phase_6_mini_gpt", 7, "07_text_generation_and_sampling", None),
    _t("complete mini-gpt package", "phase_6_mini_gpt", 8, "08_complete_minigpt_package", None),
    _t("kv cache", "phase_6_mini_gpt", 9, "09_kv_cache", None),

    # ============================================================
    # Phase 7: AI Engineer Toolkit
    # ============================================================
    _t("fine-tuning pre-trained models", "phase_7_ai_engineer_toolkit", 1, "01_finetune_gpt2_huggingface", "P16"),
    _t("fine-tuning", "phase_7_ai_engineer_toolkit", 1, "01_finetune_gpt2_huggingface", "P16"),
    _t("transfer learning", "phase_7_ai_engineer_toolkit", 1, "01_finetune_gpt2_huggingface", "P16"),
    _t("lora (low-rank adaptation)", "phase_7_ai_engineer_toolkit", 2, "02_lora_efficient_finetuning", "P16"),
    _t("lora", "phase_7_ai_engineer_toolkit", 2, "02_lora_efficient_finetuning", "P16"),
    _t("low-rank adaptation", "phase_7_ai_engineer_toolkit", 2, "02_lora_efficient_finetuning", "P16"),
    _t("rlhf (reinforcement learning from human feedback)", "phase_7_ai_engineer_toolkit", 3, "03_rlhf_three_stages", None),
    _t("simple rlhf implementation", "phase_7_ai_engineer_toolkit", 4, "04_rlhf_end_to_end_example", None),
    _t("rlhf", "phase_7_ai_engineer_toolkit", 3, "03_rlhf_three_stages", None),
    _t("alignment", "phase_7_ai_engineer_toolkit", 3, "03_rlhf_three_stages", None),
    _t("reward model", "phase_7_ai_engineer_toolkit", 5, "05_reward_model", None),
    _t("ppo", "phase_7_ai_engineer_toolkit", 6, "06_ppo_loop", None),
    _t("model serving with fastapi", "phase_7_ai_engineer_toolkit", 7, "07_fastapi_serving", "P17"),
    _t("deployment", "phase_7_ai_engineer_toolkit", 7, "07_fastapi_serving", "P17"),
    _t("fastapi", "phase_7_ai_engineer_toolkit", 7, "07_fastapi_serving", "P17"),
    _t("serving", "phase_7_ai_engineer_toolkit", 7, "07_fastapi_serving", "P17"),
    _t("dockerfile and docker-compose", "phase_7_ai_engineer_toolkit", 8, "08_docker_deployment", "P17"),
    _t("deployment commands", "phase_7_ai_engineer_toolkit", 8, "08_docker_deployment", "P17"),
    _t("production best practices", "phase_7_ai_engineer_toolkit", 9, "09_production_best_practices", None),
    _t("docker", "phase_7_ai_engineer_toolkit", 8, "08_docker_deployment", "P17"),
    _t("dockerfile", "phase_7_ai_engineer_toolkit", 8, "Dockerfile", "P17"),

    # ============================================================
    # Phase 8: Capstone & Career
    # ============================================================
    _t("ai engineering portfolio", "phase_8_capstone", 1, "01_portfolio_template", "P18"),
    _t("portfolio project ideas", "phase_8_capstone", 1, "01_portfolio_template", "P18"),
    _t("portfolio", "phase_8_capstone", 1, "01_portfolio_template", "P18"),
    _t("ai engineering interview", "phase_8_capstone", 2, "02_interview_prep_guide", None),
    _t("ai engineering career path", "phase_8_capstone", 3, "03_career_roadmap", None),
    _t("career", "phase_8_capstone", 3, "03_career_roadmap", None),
]:
    TOPIC_MAP[slug] = (folder, order, fname, project)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class CodeBlock:
    transcript_phase: int | None      # 1..6 from the original transcript
    transcript_phase_title: str | None
    nearest_header: str | None
    start_line: int
    end_line: int
    lang: str
    body: str
    docstring_title: str | None = None
    topic_slug: str | None = None
    content_hash: str = ""
    is_duplicate: bool = False
    target_folder: str | None = None
    target_filename: str | None = None
    project_id: str | None = None


FENCE_RE = re.compile(r"^```(\w*)\s*$")


def parse_transcript(text: str) -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    lines = text.splitlines()

    cur_phase: int | None = None
    cur_phase_title: str | None = None
    nearest_header: str | None = None

    inside = False
    cur_start = 0
    cur_lang = ""
    cur_body: list[str] = []
    header_at_block_start: str | None = None

    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")

        # Detect phase headers / weekly headers / generic headers when NOT inside a fence
        if not inside:
            m = PHASE_HEADER_RE.match(line)
            if m:
                cur_phase = int(m.group(1))
                cur_phase_title = m.group(2).strip()
                nearest_header = line
                continue
            m = WEEK_HEADER_RE.match(line)
            if m:
                nearest_header = line
                continue
            m = GENERIC_H_RE.match(line)
            if m:
                # only treat as nearest header if it's a # / ## / ### heading
                nearest_header = line
                continue

        m = FENCE_RE.match(line)
        if m:
            if not inside:
                inside = True
                cur_start = i
                cur_lang = (m.group(1) or "plain").lower()
                cur_body = []
                header_at_block_start = nearest_header
            else:
                inside = False
                blocks.append(
                    CodeBlock(
                        transcript_phase=cur_phase,
                        transcript_phase_title=cur_phase_title,
                        nearest_header=header_at_block_start,
                        start_line=cur_start,
                        end_line=i,
                        lang=cur_lang,
                        body="\n".join(cur_body) + "\n",
                    )
                )
            continue

        if inside:
            cur_body.append(line)

    return blocks


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------

DOCSTRING_RE = re.compile(r'^\s*"""\s*\n?(.*?)"""', re.DOTALL)


def extract_title(block: CodeBlock) -> str | None:
    """Pull a friendly title for a code block.

    Strategy:
      1. If the block starts with a triple-quoted docstring, use its first non-empty
         non-divider line.
      2. Else, fall back to the nearest preceding markdown header.
    """
    body = block.body.lstrip()
    if body.startswith('"""'):
        m = DOCSTRING_RE.search(body)
        if m:
            doc = m.group(1)
            for raw_line in doc.splitlines():
                ln = raw_line.strip()
                if not ln:
                    continue
                if set(ln) <= {"=", "-", "*", "_"}:
                    continue
                return ln

    if block.nearest_header:
        cleaned = re.sub(r"^#+\s*", "", block.nearest_header).strip()
        cleaned = re.sub(r"^(WEEK|PHASE)\s+\d+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(Day|Days)\s+\d[\d-]*\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned or None

    return None


# ---------------------------------------------------------------------------
# Topic classification
# ---------------------------------------------------------------------------

def classify(block: CodeBlock) -> tuple[str, str, str | None]:
    """Return (target_folder, target_filename_stem, project_id_or_None) for this block."""
    title = (block.docstring_title or "").lower()
    header = (block.nearest_header or "").lower()
    haystack = f"{title}\n{header}"

    # Exact slug match first
    for slug, (folder, _order, fname, project) in TOPIC_MAP.items():
        if slug in title:
            return folder, fname, project

    # Fall back to header
    for slug, (folder, _order, fname, project) in TOPIC_MAP.items():
        if slug in header:
            return folder, fname, project

    # Lang-based hints
    if block.lang in ("dockerfile", "docker"):
        return "phase_7_ai_engineer_toolkit", "Dockerfile", "P17"
    if block.lang in ("yaml", "yml"):
        return "phase_7_ai_engineer_toolkit", "deploy_config", "P17"

    # Phase fallback (just dump into the corresponding "_misc" folder)
    if block.transcript_phase:
        new_phase = TRANSCRIPT_TO_NEW_PHASE.get(block.transcript_phase, "_unsorted")
    else:
        new_phase = "_unsorted"
    return new_phase, "misc", None


# Mapping from ORIGINAL transcript phase (1..6) -> NEW curriculum phase folder
TRANSCRIPT_TO_NEW_PHASE = {
    1: "phase_1_math_engine",
    2: "phase_3_neural_networks",
    3: "phase_4_nlp_and_sequences",
    4: "phase_5_transformer",
    5: "phase_6_mini_gpt",
    6: "phase_7_ai_engineer_toolkit",
}


# ---------------------------------------------------------------------------
# Dedup + write
# ---------------------------------------------------------------------------

def normalize_for_hash(body: str) -> str:
    # Strip trailing whitespace per line so cosmetic differences don't break dedup.
    return "\n".join(line.rstrip() for line in body.splitlines()).strip()


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    text = TRANSCRIPT.read_text(encoding="utf-8")
    blocks = parse_transcript(text)
    print(f"Parsed {len(blocks)} fenced blocks from transcript.")

    # Decorate each block: title, hash, classification
    seen_hashes: dict[str, int] = {}
    counts_by_topic: dict[tuple[str, str], int] = defaultdict(int)

    skipped_trivial: list[CodeBlock] = []
    unique_blocks: list[CodeBlock] = []
    for blk in blocks:
        blk.docstring_title = extract_title(blk)
        blk.content_hash = hashlib.sha1(normalize_for_hash(blk.body).encode()).hexdigest()[:12]

        if blk.content_hash in seen_hashes:
            blk.is_duplicate = True
            continue
        seen_hashes[blk.content_hash] = blk.start_line

        # Skip tiny plain-text "Files Created / Your Skills / Final Words" summary
        # bullet-lists from the chat — they aren't code.
        if blk.lang in ("plain", "plaintext", "txt") and len(blk.body.splitlines()) < 25:
            skipped_trivial.append(blk)
            continue

        # Skip tiny shell snippets (1-2 lines) which are just `brew install` / `pip install`
        # one-liners; they're already covered as a bundle in the setup file.
        if blk.lang in ("bash", "shellscript", "shell") and len(blk.body.strip().splitlines()) <= 2:
            skipped_trivial.append(blk)
            continue

        folder, fname_stem, project = classify(blk)
        blk.target_folder = folder
        blk.project_id = project

        # Resolve filename collisions within the same folder by appending _vN
        key = (folder, fname_stem)
        counts_by_topic[key] += 1
        n = counts_by_topic[key]

        ext = LANG_EXT.get(blk.lang, "txt")
        if fname_stem == "Dockerfile":
            blk.target_filename = "Dockerfile" if n == 1 else f"Dockerfile.{n}"
        else:
            suffix = "" if n == 1 else f"_v{n}"
            blk.target_filename = f"{fname_stem}{suffix}.{ext}"

        unique_blocks.append(blk)

    print(
        f"Unique blocks after dedup: {len(unique_blocks)} "
        f"(removed {len(blocks) - len(unique_blocks) - len(skipped_trivial)} duplicates, "
        f"skipped {len(skipped_trivial)} trivial summary blocks)"
    )

    # ----- Write files -----
    for blk in unique_blocks:
        out_dir = OUT_ROOT / (blk.target_folder or "_unsorted")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (blk.target_filename or f"block_{blk.start_line}.txt")

        header_comment = (
            f"# Source: AI_Learning_Cursor lines {blk.start_line}-{blk.end_line}\n"
            f"# Original transcript phase: {blk.transcript_phase} - {blk.transcript_phase_title}\n"
            f"# Nearest header: {blk.nearest_header}\n"
            f"# Title: {blk.docstring_title}\n"
            f"#\n"
            f"# This file was extracted automatically by reference_code/_extract_from_transcript.py\n"
            f"# from the original Cursor session that produced this curriculum.\n"
            f"# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,\n"
            f"# then diff your version against this one to find blind spots.\n"
            f"\n"
        )
        if blk.lang in ("dockerfile", "docker"):
            header_comment = header_comment.replace("# Source:", "# Source:")  # keep as-is, # works in Dockerfile
        if blk.lang in ("yaml", "yml"):
            pass  # # comments are fine
        if blk.lang == "plain" or blk.lang == "plaintext" or blk.lang == "txt":
            # Plain text files: prepend a short note instead of code-style header
            header_comment = (
                f"// Source: AI_Learning_Cursor lines {blk.start_line}-{blk.end_line}\n"
                f"// Phase: {blk.transcript_phase} - {blk.transcript_phase_title}\n"
                f"// Header: {blk.nearest_header}\n"
                f"\n"
            )

        out_path.write_text(header_comment + blk.body, encoding="utf-8")

    # ----- Write extraction report -----
    lines = []
    lines.append("EXTRACTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Total blocks parsed:     {len(blocks)}")
    lines.append(f"Duplicate blocks:        {len(blocks) - len(unique_blocks)}")
    lines.append(f"Unique blocks written:   {len(unique_blocks)}")
    lines.append("")
    lines.append("Per-folder counts:")
    folder_counts: dict[str, int] = defaultdict(int)
    for blk in unique_blocks:
        folder_counts[blk.target_folder or "_unsorted"] += 1
    for folder, n in sorted(folder_counts.items()):
        lines.append(f"  {folder:40s}  {n:4d} files")
    lines.append("")
    lines.append("ALL UNIQUE BLOCKS (in transcript order):")
    lines.append("-" * 80)
    for blk in unique_blocks:
        lines.append(
            f"L{blk.start_line:5d}-L{blk.end_line:5d} "
            f"({blk.end_line - blk.start_line - 1:4d}L, lang={blk.lang:10s}) "
            f"phase={blk.transcript_phase} -> "
            f"{blk.target_folder}/{blk.target_filename}"
        )
        lines.append(f"   header : {blk.nearest_header}")
        lines.append(f"   title  : {blk.docstring_title}")
        lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {sum(folder_counts.values())} files across {len(folder_counts)} folders.")
    print(f"Audit log: {REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
