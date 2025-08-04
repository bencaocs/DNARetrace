# DNA Sequence Trace Reconstruction and Analysis Project

## Project Background

With the development of high-throughput sequencing technologies, accurate reconstruction and structural analysis of DNA sequences have become important tasks in bioinformatics. This project combines deep learning and Graph Neural Network (GNN) methods to achieve DNA sequence trace reconstruction, graph structure construction, model training, and evaluation, supporting applications such as genome assembly and sequence alignment.

---

## Directory Structure

```
GenerateData/      # Data generation and preprocessing tools
Module/            # Core algorithms, models, training, and evaluation code
TestData/          # Test datasets and examples
```

---

## Folder Details

### 1. GenerateData

- **build_graph/**  
  Scripts for converting DNA sequence data into graph structures, supporting generation and parsing of binary graph files (e.g., `.bin` format).
- **FastLabelbyKISS/**  
  Special thanks to [m5imunovic/curly-octo-train](https://github.com/m5imunovic/curly-octo-train) for sharing their work. Their implementation provided valuable reference for our project.
  This tool can be used to convert Genome data into the GFA format. Furthermore, our team has enhanced it with new edge feature calculations and can generate formats that support DGL.
- **GenomeGraphDataGeneratebyLJA/**  
  Other data generation or preprocessing tools, supporting conversion and standardization of different data formats for rapid labeling of training data.

### 2. Module

- **algorithms.py**  
  Core algorithms for DNA sequence trace reconstruction, including graph traversal and node feature extraction.
- **evaluate.py**  
  Model evaluation methods, supporting multiple metrics (accuracy, precision, recall, F1 score, MCC, etc.).
- **example_pyg_gcn.py**  
  PyTorch Geometric-based GCN model example, demonstrating how to build and train a graph neural network.
- **graph_dataset.py**  
  Dataset loading and processing module, supporting reading graph data from binary files or standard formats.
- **hyperparameters.py**  
  Hyperparameter configuration file for adjusting model structure and training parameters.
- **train.py**  
  Main training script, supporting multi-model training, automatic checkpoint saving, and logging.
- **utils.py**  
  Common utility functions, such as data format conversion and logging.
- **checkpoints/**  
  Directory for saving model weights during training.
- **dataset/**  
  Directory for processed dataset files.
- **layers/**  
  Neural network layer definitions, supporting custom GNN layer structures.
- **models/**  
  Definitions of different model structures for easy extension and comparative experiments.

### 3. TestData

- **ONT/data/graph-kmer15/**  
  Binary graph files for testing (e.g., `graph_kmer_15_0.bin`), used for model evaluation and algorithm validation.
- **ONT/data/seq/**  
  DNA sequence files for testing, supporting sequences of different lengths and complexities.

---

## Usage

### Environment Setup

It is recommended to use Anaconda or Miniconda to manage the environment:

```bash
conda env create -f GenerateData/curly-octo-train/environment.yaml
conda activate dna-trace-env
pip install -r GenerateData/curly-octo-train/requirements.txt
```

### Data Generation and Preprocessing

1. Use tools such as pbsim3 to generate synthetic DNA sequence data:
    ```bash
    cd GenerateData/curly-octo-train
    ./pbsim3 [parameter configuration]
    ```
2. Run the `build_graph` script to convert sequence data into graph structures:
    ```bash
    python GenerateData/build_graph/build_graph.py --input seq.fasta --output graph.bin
    ```

### Model Training and Evaluation

1. Configure hyperparameters (can be modified in `Module/hyperparameters.py`).
2. Run the training script:
    ```bash
    python Module/train.py --data TestData/ONT/data/graph-kmer15/graph_kmer_15_0.bin --epochs 100 --batch_size 32
    ```
3. The model will be automatically saved to the `Module/checkpoints/` folder during training.
4. Evaluate model performance:
    ```bash
    python Module/evaluate.py --model Module/checkpoints/best_model.pth --test_data TestData/ONT/data/seq/test_seq.fasta
    ```

### Testing and Result Analysis

- Test data can be placed in the `TestData/ONT/data/graph-kmer15/` and `TestData/ONT/data/seq/` folders.
- The evaluation script supports detailed metrics output and visualization of results.

---

## Main Features and Highlights

- **Data Generation and Preprocessing**: Supports multi-format processing of synthetic and real data.
- **Graph Structure Construction**: Efficiently converts DNA sequences into graph structures for further analysis.
- **Multi-model Support**: Integrates various GNN models for flexible extension and comparative experiments.
- **Automated Training and Evaluation**: One-click training, automatic model saving, and detailed evaluation reports.
- **Visualization Analysis**: Supports visualization of training processes and results.

---

## Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **MCC (Matthews Correlation Coefficient)**

---

## Reference Tools

- [pbsim3](https://github.com/pfaucon/PBSIM-PacBio-Simulator)
- [KISS](https://github.com/jhhung/kISS)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)