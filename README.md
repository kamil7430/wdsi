# WDSI Repository

**WDSI** - Wprowadzenie do Sztucznej Inteligencji (Introduction to Artificial Intelligence)

This repository contains three main projects from the WDSI program.

---

## Part 1: Machine Learning Fundamentals

Introduction to core supervised learning techniques.

### Contents
- **Classification**: Notebook-based exploration of classification algorithms with training data
- **Regression**: Notebook-based regression analysis with training datasets

### Tech Stack
- scikit-learn, pandas, numpy, seaborn, matplotlib, imbalanced-learn

---

## Part 2: Reinforcement Learning

Hands-on labs and projects implementing reinforcement learning algorithms.

### Lab Section
Three guided labs working with "slippery world" environments:
- **Lab 1-3**: Interactive exercises exploring RL concepts in controlled environments

### Project Section
Implementing and comparing different RL algorithms on two environments (slippery and dangerous worlds):
- **Algorithms implemented**:
  - Q-Learning (learns value function from experience)
  - SARSA (on-policy temporal difference learning)
  - Dyna-Q (integrates model-free and model-based learning)
  - Value Iteration (dynamic programming approach)
  
- **Key files**:
  - `algorithms.py`: Core RL algorithm implementations
  - `experiments.py`: Experiment runners and evaluations
  - Individual notebooks for each algorithm on each world variant
  - `wdsi_rl.pdf`: Theoretical guide and documentation

---

## Part 3: GPU Recommendation System

Collaborative project building intelligent GPU selection using NLP approaches. Dataset: 40 years of GPU specs and pricing (1986-2026).

### Kamilos - Rule-Based NLP Extraction
**Goal**: Parse natural language GPU queries into structured requirements

**Approach**:
- Uses spaCy NLP library with Polish language support (`pl_core_news_sm`)
- Custom entity recognition patterns for GPU specifications

**Key components**:
- **data.py**: 
  - Loads and cleans raw GPU dataset (Brand, Model, VRAM, Architecture, Performance, TDP, Bandwidth, Price)
  - Parses memory values with unit conversion (KB/MB/GB/TB)
  - Extracts numeric specs (TDP watts, FP32 TFLOPS)
  - Computes rental price per hour based on power, compute, and memory costs
  
- **model.py**:
  - Polish NLP extraction pipeline
  - Recognizes entities: VRAM specs, GPU brand (NVIDIA/AMD/Intel), budget constraints
  - Handles written numbers in Polish ("osiem" = 8, "dwa tysiące" = 2000)
  - Converts user queries like "Chcę 16GB taniej" → `{min_vram_gb: 16, is_cheap: True}`
  
- **rank.py**: Ranks GPU options based on extracted criteria
- **main.py**: Interactive CLI for GPU recommendations
- **tests.py**: Comprehensive test suite for extraction logic

**Tech**: spaCy, pandas, numpy

### Pablo - Semantic Embeddings
**Goal**: Match user queries to GPUs using semantic similarity

**Approaches**:

1. **BERT-based semantic search** (bert.py):
   - Uses sentence transformers (all-MiniLM-L6-v2 model)
   - Converts each GPU into natural text description with specs and labels (cheap/expensive, small/large VRAM)
   - Encodes both GPU descriptions and user query into embeddings
   - Finds GPU with highest semantic similarity to query
   - Example: "64 GB VRAM, price 1500 cents/hr, NVIDIA" → finds best match via embedding similarity

2. **spaCy dependency parsing** (our_spacy.py):
   - English syntax analysis
   - Extracts dependencies between words (modifiers, constraints)
   - Filters noise (pronouns, auxiliary verbs)
   - Example processing of "I want not greater than 64 gb vram..."

**Tech**: sentence-transformers, spaCy, pandas, numpy

### Running the systems
```bash
# Kamilos (Polish rule-based)
cd Part3/Kamilos
pip install -r requirements.txt
python main.py

# Pablo (semantic & parsing)
cd Part3/Pablo
pip install -r requirements.txt
python bert.py
python our_spacy.py
```

### Dataset
- **gpu_1986-2026.csv** (2MB): Specifications for thousands of GPU models across 40 years
- Shared between both approaches for comparison

---

## Project Summary

| Part | Focus | Techniques |
|------|-------|-----------|
| 1 | ML Basics | Classification, Regression |
| 2 | RL Algorithms | Q-Learning, SARSA, Dyna-Q, Value Iteration |
| 3 | NLP + Application | Rule-based extraction (spaCy), Semantic embeddings (BERT), Dependency parsing |
