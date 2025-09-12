# Modanovo

**Modanovo** is a de novo peptide sequencing tool for **post-translationally modified (PTM) peptides**, built on top of [Casanovo (v4.0.0)](https://github.com/Noble-Lab/casanovo/tree/v4.0.0).

---

## Installation

We recommend using a fresh [conda](https://docs.conda.io/) environment with Python 3.10.

```bash
conda create --name modanovo-env python=3.10
conda activate modanovo-env
```

Install the dependencies:

1) **PyTorch** (pick the command matching your CUDA/CPU setup from the PyTorch site; generic example):

```bash
pip3 install torch
```

2) **Depthcharge-MS** (pinned commit):

```bash
pip install git+https://github.com/wfondrie/depthcharge.git@bd2861f
```

3) **Clone this repository**:

```bash
git clone https://github.com/gagneurlab/Modanovo.git
cd modanovo
```

4) **Install Modanovo**:

```bash
pip install .
```

For development (editable install):

```bash
pip install -e .
```

---

## Usage

Modanovo supports three modes: **training**, **evaluation**, and **inference**.

### Training

Train a new model from scratch:

```bash
modanovo train -c <config_path> -p <val_paths> <train_paths>
```

Fine-tune from pretrained Casanovo weights:

```bash
modanovo train -c <config_path> -m <model_path> -p <val_paths> <train_paths>
```

Where `<model_path>` points to the pretrained **Casanovo v4.0.0** weights.

### Evaluation

Evaluate a trained model on validation/test spectra:

```bash
modanovo evaluate -c <config_path> -m <model_path> -p <val_paths>
```

### Inference

Run Modanovo in inference mode:

```bash
modanovo sequence -c <config_path> -m <model_path> -o <out_path>
```

This writes peptide sequence predictions in **`.mzTab`** format.

---

## Quickstart (example)

Assuming you’ve installed Modanovo and have a model checkpoint:

```bash
# from the repo root
modanovo inference \
  -c modanovo/config.yaml \
  -m path/to/casanovo_or_modanovo_weights.ckpt \
  -o outputs/predictions.mztab
```

Make sure that the defined residues are compatible with the model weights. Leaving the config entry `expanded_residues` in the configuration file empty uses Casanovo's tokens. By default, fine-tuning residues are those from the **MULTI-PTM** dataset in **PROSPECT-PTM**.

---

## Example data & configs

- Default configuration: [`modanovo/config.yaml`](./modanovo/config.yaml)  
- Example spectra file: [`data_utils/example_data.mgf`](./data_utils/example_data.mgf)  
- Train/val/test splits used during development: [`https://huggingface.co/datasets/gagneurlab/Modanovo-development-dataset`](https://huggingface.co/datasets/gagneurlab/Modanovo-development-dataset)
- Model weights for the model fine-tuned to cover 19 amino acid-PTM combinations: [`https://huggingface.co/gagneurlab/Modanovo-model`](https://huggingface.co/gagneurlab/Modanovo-model)
---

## Compatibility

- Compatible with **Casanovo v4.0.0** weights and formats.

---

## References

- **Casanovo**: _Yilmaz, Melih, William E Fondrie, Wout Bittremieux, et al. 2024. “Sequence-to-Sequence Translation from Mass Spectra to Peptides with a Transformer Model.” Nature Communications 15 (1): 6427.
  
- **PROSPECT-PTM**: Gabriel, Wassim, Omar Shouman, Ayla Schroeder, Florian Boessl, and Mathias Wilhelm. 2024. “PROSPECT PTMs: Rich Labeled Tandem Mass Spectrometry Dataset of Modified Peptides for Machine Learning in Proteomics.” Advances in Neural Information Processing Systems 37.

---

## Citation

If you use **Modanovo** in your research, please cite:

```
FIXME
```
