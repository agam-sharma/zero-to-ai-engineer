"""
One-shot fix-up that gives the v2/v3/v4/v5 collision files precise filenames
based on what each one actually contains (we identified each block by reading
its source-line range from AI_Learning_Cursor).

Run once after _extract_from_transcript.py:
    python3 reference_code/_finalize_filenames.py
"""

from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (current_path_relative_to_ROOT, new_path_relative_to_ROOT)
RENAMES: list[tuple[str, str]] = [
    # ---- The 5 "Core NumPy Concepts" sub-blocks were collapsed to v1..v5 ----
    # v1 (L391-409): creating arrays + initialization patterns -> keep, but rename
    ("phase_1_math_engine/02_core_numpy_concepts.py",
     "phase_1_math_engine/02_creating_arrays_and_init.py"),
    # v2 (L412-428): dot product (already had a mapping target, this is the actual content)
    ("phase_1_math_engine/02_core_numpy_concepts_v2.py",
     "phase_1_math_engine/03_dot_product_drills.py"),
    # v3 (L431-459): matrix multiplication
    ("phase_1_math_engine/02_core_numpy_concepts_v3.py",
     "phase_1_math_engine/04_matrix_multiplication.py"),
    # v4 (L462-481): broadcasting
    ("phase_1_math_engine/02_core_numpy_concepts_v4.py",
     "phase_1_math_engine/05_broadcasting.py"),
    # v5 (L484-504): reshape patterns
    ("phase_1_math_engine/02_core_numpy_concepts_v5.py",
     "phase_1_math_engine/05a_reshape_patterns.py"),

    # ---- The 4 "Key Python Patterns" sub-blocks were collapsed to v1..v4 ----
    # v1 (L642-686): defines a Neuron CLASS  ->  class pattern + neuron in one
    ("phase_1_math_engine/07_python_patterns_for_ai.py",
     "phase_1_math_engine/07_python_pattern_class_neuron.py"),
    # v2 (L689-711): list comprehension / dataset preprocessing
    ("phase_1_math_engine/07_python_patterns_for_ai_v2.py",
     "phase_1_math_engine/07a_python_pattern_list_comprehensions.py"),
    # v3 (L714-733): batch generator function (yield)
    ("phase_1_math_engine/07_python_patterns_for_ai_v3.py",
     "phase_1_math_engine/07b_python_pattern_batch_generator.py"),
    # v4 (L736-770): context manager (gradient tracking)
    ("phase_1_math_engine/07_python_patterns_for_ai_v4.py",
     "phase_1_math_engine/07c_python_pattern_context_manager.py"),

    # ---- The single _unsorted/misc.py = matrices intro (L1060-1100) ----
    ("_unsorted/misc.py",
     "phase_1_math_engine/12_matrices_as_transformations.py"),

    # ---- Other v2 collisions (each is genuinely distinct) ----
    # Phase 1: gradient_descent_v2 = the 2D extension assignment
    ("phase_1_math_engine/18_gradient_descent_from_scratch_v2.py",
     "phase_1_math_engine/18a_gradient_descent_2d_assignment.py"),

    # Phase 0: the python "PyTorch MPS check" that landed under setup
    ("phase_0_environment/01_mac_m3_pro_setup_commands_v2.py",
     "phase_0_environment/02_pytorch_mps_check.py"),

    # Phase 3: the second "MNIST classifier" file = data augmentation variant
    ("phase_3_neural_networks/07_mnist_classifier_v2.py",
     "phase_3_neural_networks/07a_mnist_classifier_with_augmentation.py"),

    # Phase 7: fastapi_serving_v2 = deployment commands (different content)
    ("phase_7_ai_engineer_toolkit/07_fastapi_serving_v2.py",
     "phase_7_ai_engineer_toolkit/07a_deployment_commands.py"),
]


def main() -> None:
    moved = 0
    for src_rel, dst_rel in RENAMES:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            print(f"  SKIP (missing): {src_rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"  SKIP (target exists): {dst_rel}")
            continue
        shutil.move(str(src), str(dst))
        print(f"  {src_rel}  ->  {dst_rel}")
        moved += 1

    # Clean up empty _unsorted folder if everything moved out
    unsorted_dir = ROOT / "_unsorted"
    if unsorted_dir.exists() and not any(unsorted_dir.iterdir()):
        unsorted_dir.rmdir()
        print(f"  Removed empty _unsorted/")

    print(f"\nMoved {moved} file(s).")


if __name__ == "__main__":
    main()
