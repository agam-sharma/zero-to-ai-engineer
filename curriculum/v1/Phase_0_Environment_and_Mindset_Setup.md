# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: ENVIRONMENT & MINDSET SETUP (Week 0)
# ═══════════════════════════════════════════════════════════════════════
# Zero to GPT — 6-Month AI Engineering Masterclass
# Tailored for: Senior Salesforce Engineer → AI Engineer Transition
# Hardware: Mac M3 Pro (Apple Silicon, 18GB Unified Memory, MPS GPU)
# ═══════════════════════════════════════════════════════════════════════

---

## 🔗 The Salesforce Analogy

> **Setting up your AI dev environment is like setting up a brand-new Salesforce DX project.**
>
> In Salesforce, before you write your first line of Apex, you need:
> - `sfdx` CLI installed
> - VS Code with Salesforce Extension Pack
> - A properly configured `sfdx-project.json`
> - A scratch org or sandbox
> - A working `package.xml` manifest
>
> Skip ANY of these? You're debugging config issues for hours instead of building features.
>
> AI engineering is identical. Before you can train your first neuron, you need:
> - Python environment (like your CLI)
> - PyTorch with MPS GPU support (like your org connection)
> - Jupyter notebooks + VS Code (like Developer Console + VS Code)
> - A clean project structure with Git (like your SFDX project structure)
>
> **We set this up RIGHT, ONCE, so you NEVER fight it again.**

---

## 📖 THEORY: Why Environment Setup Matters in AI

### The #1 Reason Beginners Quit AI

It's not the math. It's not the code. It's **environment issues**.

```
Common Beginner Horror Stories:
├── "Which Python version do I use?" (3.8? 3.10? 3.11? System Python?)
├── "pip install torch fails with cryptic C++ errors"
├── "My training is taking 45 minutes for a simple model" (running on CPU, not GPU)
├── "I have 3 different Python installations fighting each other"
└── "import numpy says 'module not found' even though I installed it"
```

You will **never** have these problems because we're setting up a **clean, isolated conda environment** with everything pre-configured for your M3 Pro's GPU.

### Understanding Your M3 Pro's AI Capabilities

Your Mac M3 Pro is a genuinely powerful machine for AI development:

```
╔══════════════════════════════════════════════════════════════════╗
║  MAC M3 PRO — AI HARDWARE PROFILE                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  CPU:  11-core (5 performance + 6 efficiency)                    ║
║  GPU:  14-core or 18-core (Apple Silicon integrated)             ║
║  RAM:  18GB Unified Memory (shared between CPU and GPU)          ║
║  NPU:  16-core Neural Engine (for inference optimization)        ║
║                                                                  ║
║  WHAT THIS MEANS FOR AI:                                         ║
║  ✅ Can train models up to ~100-120M parameters comfortably      ║
║  ✅ GPU training via Metal Performance Shaders (MPS)             ║
║  ✅ Unified memory = no CPU↔GPU data transfer bottleneck         ║
║  ✅ Can run quantized 7B models (LLaMA, Mistral) for inference   ║
║  ⚠️  Cannot match NVIDIA A100/H100 for very large models         ║
║  ⚠️  ~18GB RAM limits maximum model + data size during training  ║
║                                                                  ║
║  BOTTOM LINE: You can train a model BIGGER than GPT-1 (117M     ║
║  params) entirely on this machine. That's more than enough.      ║
╚══════════════════════════════════════════════════════════════════╝
```

### What is MPS (Metal Performance Shaders)?

In the NVIDIA/CUDA world, GPU-accelerated AI training uses CUDA cores. Apple's equivalent is **MPS (Metal Performance Shaders)** — Apple's framework for running compute-intensive operations on the GPU.

**Without MPS:** Your PyTorch code runs on CPU only → training a model takes 30 minutes.
**With MPS:** Same code runs on GPU → same model trains in 3 minutes.

That's a **10x speedup** — and it requires changing just ONE line of code:

```python
# WITHOUT MPS (slow — CPU only)
device = torch.device("cpu")

# WITH MPS (fast — GPU accelerated)
device = torch.device("mps")
```

Every single project in this course will use MPS. We'll set it up now so it's automatic.

---

## 🛠️ STEP-BY-STEP SETUP GUIDE

### Step 0.1: Install Xcode Command-Line Tools

MPS requires Apple's development tools. Open Terminal and run:

```bash
xcode-select --install
```

A popup will appear — click "Install" and wait (this downloads ~2.5GB).

**Verify:**
```bash
xcode-select -p
# Should output: /Library/Developer/CommandLineTools
```

**Salesforce Parallel:** This is like installing the `sfdx` CLI — it's the foundational tool everything else depends on.

---

### Step 0.2: Install Miniconda (Python Environment Manager)

**Why Miniconda and not just "python3"?**

Your Mac comes with a system Python. **Never use it for AI projects.** It's like using a production Salesforce org for development — you'll corrupt it eventually.

Miniconda creates **isolated virtual environments** — each project gets its own Python version and packages, completely independent of each other.

```bash
# Download Miniconda for Apple Silicon (ARM64)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install it (follow the prompts, say "yes" to everything)
sh Miniconda3-latest-MacOSX-arm64.sh

# IMPORTANT: Close and reopen your terminal after installation
# Or run:
source ~/.zshrc
```

**Verify Miniconda is installed:**
```bash
conda --version
# Should output something like: conda 24.x.x
```

**Salesforce Parallel:** Miniconda is like Salesforce DX scratch orgs — disposable, isolated environments where you can experiment without affecting anything else.

---

### Step 0.3: Create Your AI Environment

```bash
# Create a new conda environment named "ai-mastery" with Python 3.11
conda create -n ai-mastery python=3.11 -y

# Activate it (you'll need to do this every time you open a new terminal)
conda activate ai-mastery

# Verify you're in the right environment
which python
# Should show: /Users/agam.sharma/miniconda3/envs/ai-mastery/bin/python

python --version
# Should show: Python 3.11.x
```

**Why Python 3.11?**
- Python 3.11 is 10-25% faster than 3.10 for CPU-bound operations
- Full compatibility with PyTorch 2.10.0, NumPy, and all libraries we'll use
- Well-tested, stable, and widely supported

**Pro Tip:** Add this to your `~/.zshrc` so the environment auto-activates:
```bash
echo 'conda activate ai-mastery' >> ~/.zshrc
```

---

### Step 0.4: Install PyTorch with MPS Support

This is the most important installation. PyTorch is the deep learning framework we'll use for the entire course — it's the "Salesforce Platform" of AI.

```bash
# Make sure you're in the ai-mastery environment
conda activate ai-mastery

# Install PyTorch (stable release — currently 2.10.0)
pip3 install torch torchvision torchaudio

# Verify installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
# Should show: PyTorch version: 2.x.x
```

**Now verify MPS GPU acceleration:**

```python
# Save this as verify_mps.py and run it
import torch
import time

print("=" * 60)
print("🔍 M3 PRO AI ENVIRONMENT VERIFICATION")
print("=" * 60)

# Check PyTorch version
print(f"\n📦 PyTorch version: {torch.__version__}")

# Check MPS availability
print(f"\n🖥️  MPS (Metal GPU) available: {torch.backends.mps.is_available()}")
print(f"🖥️  MPS built into PyTorch:    {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    
    # Create a tensor on MPS
    x = torch.ones(5, device=mps_device)
    print(f"\n✅ Tensor on MPS device: {x}")
    print(f"   Device: {x.device}")
    
    # Speed comparison: CPU vs MPS
    print("\n" + "=" * 60)
    print("⚡ SPEED TEST: CPU vs MPS (GPU)")
    print("=" * 60)
    
    size = 4096  # 4096 x 4096 matrix multiplication
    
    # CPU test
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    start = time.time()
    c_cpu = a_cpu @ b_cpu  # Matrix multiply on CPU
    cpu_time = time.time() - start
    print(f"\n🐢 CPU: {size}x{size} matmul = {cpu_time:.4f} seconds")
    
    # MPS test
    a_mps = torch.randn(size, size, device=mps_device)
    b_mps = torch.randn(size, size, device=mps_device)
    
    # Warm up the GPU
    _ = a_mps @ b_mps
    torch.mps.synchronize()
    
    start = time.time()
    c_mps = a_mps @ b_mps  # Matrix multiply on GPU
    torch.mps.synchronize()  # Wait for GPU to finish
    mps_time = time.time() - start
    print(f"🚀 MPS: {size}x{size} matmul = {mps_time:.4f} seconds")
    
    speedup = cpu_time / mps_time
    print(f"\n⚡ MPS is {speedup:.1f}x faster than CPU!")
    
    print("\n" + "=" * 60)
    print("✅ YOUR M3 PRO IS READY FOR AI TRAINING!")
    print("=" * 60)
else:
    print("\n❌ MPS not available. Check your macOS version (need 12.3+)")
    print("   and PyTorch installation.")
```

**Expected output (approximately):**
```
==============================================================
🔍 M3 PRO AI ENVIRONMENT VERIFICATION
==============================================================

📦 PyTorch version: 2.10.0

🖥️  MPS (Metal GPU) available: True
🖥️  MPS built into PyTorch:    True

✅ Tensor on MPS device: tensor([1., 1., 1., 1., 1.], device='mps:0')
   Device: mps:0

==============================================================
⚡ SPEED TEST: CPU vs MPS (GPU)
==============================================================

🐢 CPU: 4096x4096 matmul = 0.7234 seconds
🚀 MPS: 4096x4096 matmul = 0.0521 seconds

⚡ MPS is 13.9x faster than CPU!

==============================================================
✅ YOUR M3 PRO IS READY FOR AI TRAINING!
==============================================================
```

> **If you see this output, congratulations — your GPU is working.** Every training run in this course will leverage this speedup.

---

### Step 0.5: Install All Required Libraries

```bash
conda activate ai-mastery

# === CORE SCIENTIFIC COMPUTING ===
pip3 install numpy pandas matplotlib seaborn

# === MACHINE LEARNING ===
pip3 install scikit-learn

# === JUPYTER NOTEBOOKS ===
pip3 install jupyterlab ipywidgets

# === HUGGING FACE ECOSYSTEM (for Phase 6) ===
pip3 install transformers datasets tokenizers accelerate

# === EXPERIMENT TRACKING ===
pip3 install wandb tqdm

# === TOKENIZATION (for Karpathy's GPT Tokenizer video) ===
pip3 install tiktoken

# === VISUALIZATION ===
pip3 install plotly
```

**Verify all imports work:**

```python
# Save as verify_imports.py and run it
imports = [
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("sklearn", "Scikit-learn"),
    ("transformers", "HuggingFace Transformers"),
    ("datasets", "HuggingFace Datasets"),
    ("tiktoken", "OpenAI Tiktoken"),
    ("wandb", "Weights & Biases"),
    ("tqdm", "Progress Bars"),
]

print("📦 Library Verification")
print("=" * 50)
all_good = True
for module, name in imports:
    try:
        __import__(module)
        print(f"  ✅ {name:30s} — installed")
    except ImportError:
        print(f"  ❌ {name:30s} — MISSING!")
        all_good = False

if all_good:
    print("\n🎉 All libraries installed successfully!")
else:
    print("\n⚠️  Some libraries are missing. Run the pip install commands above.")
```

---

### Step 0.6: VS Code Setup for AI Development

You're already a VS Code pro from Salesforce. Here are the AI-specific extensions:

**Install these VS Code extensions:**

| Extension | Why |
|-----------|-----|
| **Python** (Microsoft) | Python language support, IntelliSense, debugging |
| **Jupyter** (Microsoft) | Run notebooks directly in VS Code |
| **Pylance** (Microsoft) | Fast Python type checking and autocomplete |
| **Python Indent** | Correct Python indentation (crucial — Python is whitespace-sensitive) |
| **Rainbow CSV** | View datasets in CSV format with color coding |

**VS Code Settings for AI Development:**

Add these to your VS Code `settings.json`:
```json
{
    "python.defaultInterpreterPath": "~/miniconda3/envs/ai-mastery/bin/python",
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "notebook.output.textLineLimit": 100,
    "notebook.output.scrolling": true,
    "editor.rulers": [88],
    "python.analysis.typeCheckingMode": "basic"
}
```

---

### Step 0.7: Project Structure & Git

```bash
# Create the project directory structure
mkdir -p ~/zero-to-gpt/{notebooks,src,models,data,experiments,notes}

# Initialize Git
cd ~/zero-to-gpt
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/

# Jupyter
.ipynb_checkpoints/

# Data files (too large for Git)
data/*.csv
data/*.txt
data/*.bin
data/*.pkl
*.pt
*.pth

# Models (too large for Git)
models/*.pt
models/*.bin

# Environment
.env
.venv/

# OS
.DS_Store
Thumbs.db

# Weights & Biases
wandb/

# IDE
.vscode/
.idea/
EOF

# Create README
cat > README.md << 'EOF'
# 🚀 Zero to GPT — My AI Engineering Journey

A 6-month hands-on journey from zero to building a GPT from scratch.

## Structure
- `notebooks/` — Jupyter notebooks for each week's exercises
- `src/` — Python modules and reusable code
- `models/` — Trained model checkpoints
- `data/` — Training datasets
- `experiments/` — Experiment logs and results
- `notes/` — Personal notes and reflections

## Hardware
- Mac M3 Pro (Apple Silicon, MPS GPU acceleration)
- PyTorch with Metal Performance Shaders

## Course
Based on the "Zero to GPT" 6-Month Masterclass curriculum.
EOF

# First commit
git add .
git commit -m "Initial project setup: Zero to GPT learning journey"

echo "✅ Project structure created at ~/zero-to-gpt"
```

**Salesforce Parallel:** This is exactly like your SFDX project structure:
```
Salesforce DX Project          →    AI Project
├── force-app/main/default/    →    ├── src/
│   ├── classes/               →    │   ├── models/
│   ├── lwc/                   →    │   ├── layers/
│   └── triggers/              →    │   └── utils/
├── scripts/                   →    ├── notebooks/
├── manifest/package.xml       →    ├── requirements.txt
├── sfdx-project.json          →    ├── pyproject.toml
└── .gitignore                 →    └── .gitignore
```

---

## 📖 THE MINDSET SHIFT: Deterministic → Probabilistic

This is the most important section of Phase 0. Read it carefully.

### In Salesforce (Deterministic Programming):

```apex
// You write RULES. The code does EXACTLY what you say.
public class AccountTriggerHandler {
    public static void beforeInsert(List<Account> newAccounts) {
        for (Account acc : newAccounts) {
            if (acc.Industry == 'Technology') {
                acc.Rating = 'Hot';   // This ALWAYS happens. 100% of the time.
            }
        }
    }
}
```

- Input → Output is **perfectly predictable**
- Same input ALWAYS produces same output
- A bug means your LOGIC is wrong
- "Correct" means it works for ALL valid inputs

### In AI (Probabilistic Programming):

```python
# You provide DATA. The model LEARNS patterns from it.
# Then it makes PREDICTIONS — which are PROBABILISTIC, not certain.

model = GPT(vocab_size=50000, n_layers=6, d_model=384)
model.train(shakespeare_text)

# Ask the model to generate text:
output = model.generate("To be or not to be, that is the")
# Output might be: "question" (85% confident)
#            or:   "problem"  (10% confident)
#            or:   "answer"   (5% confident)
```

- Input → Output is **probabilistic** (there's a distribution of possible answers)
- Same input can produce DIFFERENT outputs (due to sampling)
- A "bug" might be: wrong data, wrong architecture, not enough training, bad hyperparameters
- "Correct" means it works for **most** inputs (95% accuracy is a GREAT model)

### The Key Differences:

| Concept | Salesforce (Deterministic) | AI (Probabilistic) |
|---------|---------------------------|---------------------|
| **Logic** | You write explicit rules | Model learns rules from data |
| **Testing** | Assert exact equality: `assertEquals(expected, actual)` | Assert approximate: `accuracy > 0.95` |
| **Debugging** | Read the stack trace, find the bug | Visualize loss curves, check data quality, tune hyperparameters |
| **Deployment** | Deploy code → same behavior everywhere | Deploy model → behavior depends on input distribution |
| **"Correct"** | Works for ALL inputs | Works for MOST inputs |
| **Errors** | Exceptions, null pointers | Wrong predictions, hallucinations |
| **Improvement** | Better algorithms, better logic | More data, better architecture, more training |

### The 5 Mental Model Shifts You Must Make:

1. **From "exact" to "approximate":** A model that's 97% accurate is excellent. You'll never get 100%. That last 3% might cost 10x more compute and data than the first 97%.

2. **From "debugging code" to "debugging data":** In AI, most problems are data problems, not code problems. Bad training data → bad model, no matter how good your code is.

3. **From "writing logic" to "designing experiments":** You don't write the logic — the model learns it. Your job is to design the training setup (architecture, data, hyperparameters) so the model CAN learn.

4. **From "binary success" to "metrics on a spectrum":** In Salesforce, your test either passes or fails. In AI, your model has a loss of 2.3 today and 1.8 tomorrow — it's continuously improving.

5. **From "one right answer" to "a distribution of answers":** When GPT writes "The cat sat on the ___", there's no single right answer. "mat", "chair", "floor", "table" are all valid. The model learns a probability distribution over all possible words.

---

## 📚 REQUIRED READING FOR WEEK 0

### Must Read (Before Moving to Phase 1):

| # | Resource | Time | What You'll Learn |
|---|----------|------|-------------------|
| 1 | **[Andrej Karpathy: Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35)** | 15 min | The philosophical foundation: why AI is a new paradigm of programming, not just a tool. Karpathy argues that neural networks are a fundamentally different way to write software — you don't write the program, you design the training process and the network learns the program. |
| 2 | **[Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)** | 20 min | Practical wisdom from one of the best AI engineers. Read this now to plant seeds — you'll re-read it in Week 8 and it will click much deeper. |
| 3 | **[3Blue1Brown: But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)** | 19 min video | The absolute best visual introduction to neural networks. Watch this even if you think you know what a neural network is. Grant Sanderson's visualizations will give you intuition that no textbook can. |

### Optional But Excellent:

| # | Resource | Time | What You'll Learn |
|---|----------|------|-------------------|
| 4 | **[Jay Alammar: A Visual Intro to NumPy](https://jalammar.github.io/visual-numpy/)** | 15 min | Visual guide to NumPy — the library you'll use every single day. |
| 5 | **[Google's Machine Learning Crash Course: Framing](https://developers.google.com/machine-learning/crash-course/framing/video-lecture)** | 5 min video | Google's intro to how to "frame" a problem as an ML problem. |
| 6 | **[fast.ai: Practical Deep Learning — Lesson 1](https://course.fast.ai/Lessons/lesson1.html)** | 90 min video | Jeremy Howard's top-down approach. Watch if you want a preview of the full journey. |

---

## ✍️ HOMEWORK: Week 0 Assignments

### Assignment 0.1: Environment Verification (30 minutes)
Run the `verify_mps.py` and `verify_imports.py` scripts above. Paste the output into a file called `notes/week0_setup_verification.txt`. If anything fails, debug it before moving on.

### Assignment 0.2: First Notebook (30 minutes)
Create a Jupyter notebook called `notebooks/00_hello_ai.ipynb` with:
1. A markdown cell with your name and date
2. A code cell that imports torch and prints the version
3. A code cell that creates a tensor on MPS and does a matrix multiply
4. A code cell that uses matplotlib to plot `y = x²` for x in [-5, 5]

### Assignment 0.3: The Reflection (45 minutes)
After reading Karpathy's "Software 2.0" blog post, write a 1-page reflection answering:
1. What are 3 ways AI engineering differs from Salesforce engineering?
2. What Salesforce skills will transfer directly to AI? (Hint: there are more than you think)
3. What's one thing about AI that excites you? One thing that worries you?

Save this as `notes/week0_reflection.md`.

### Assignment 0.4: Speed Test (15 minutes)
Run the GPU speed test with different matrix sizes: 1024, 2048, 4096, 8192. Record the CPU vs MPS times in a table. This gives you a feel for when GPU matters (hint: it matters more as matrices get bigger — which is exactly what happens in real neural networks).

```python
import torch
import time

sizes = [1024, 2048, 4096, 8192]

print(f"{'Size':>8} | {'CPU (sec)':>10} | {'MPS (sec)':>10} | {'Speedup':>8}")
print("-" * 50)

for size in sizes:
    # CPU
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start
    
    # MPS
    a_mps = torch.randn(size, size, device="mps")
    b_mps = torch.randn(size, size, device="mps")
    _ = a_mps @ b_mps  # warmup
    torch.mps.synchronize()
    start = time.time()
    _ = a_mps @ b_mps
    torch.mps.synchronize()
    mps_time = time.time() - start
    
    print(f"{size:>8} | {cpu_time:>10.4f} | {mps_time:>10.4f} | {cpu_time/mps_time:>7.1f}x")
```

---

## ✅ PHASE 0 COMPLETION CHECKLIST

Before moving to Phase 1, verify ALL of the following:

- [ ] Xcode CLI tools installed (`xcode-select -p` works)
- [ ] Miniconda installed (`conda --version` works)
- [ ] `ai-mastery` environment created and activated
- [ ] Python 3.11 confirmed (`python --version`)
- [ ] PyTorch installed (`import torch` works)
- [ ] MPS GPU working (`torch.backends.mps.is_available()` returns `True`)
- [ ] MPS speed test shows 5x+ speedup over CPU
- [ ] All libraries installed (verify_imports.py passes)
- [ ] VS Code configured with Python + Jupyter extensions
- [ ] Project directory structure created with Git
- [ ] Read Karpathy's "Software 2.0" blog post
- [ ] Watched 3Blue1Brown "What is a Neural Network?" video
- [ ] Completed the reflection assignment
- [ ] Created `00_hello_ai.ipynb` notebook

**Once all boxes are checked, you're ready for Phase 1: The Math Engine. 🧮**

---

> **Time Investment:** ~6-8 hours over 2-3 days
>
> **Next:** [Phase 1: The Math Engine (Weeks 1–6)](./Phase_1_The_Math_Engine.md) — Linear algebra + calculus + **probability & information theory** (new). The mathematical foundations that power every AI model and every loss function.
