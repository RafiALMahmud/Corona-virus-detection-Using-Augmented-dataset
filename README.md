# COVID-19 Radiography — CycleGAN Augmentation & Classification Pipeline

**A lightweight end-to-end pipeline that augments chest X‑ray images using a CycleGAN, trains classifiers (MobileNetV2, SimpleCNN, ResNet50), and evaluates the effect of synthetic augmentation and knowledge distillation.**

---

## Project overview

This repository contains a research/experimental pipeline implemented in TensorFlow/Keras for augmenting a COVID-19 chest X‑ray dataset with synthetic images produced by a lightweight CycleGAN, training several image classifiers on both the real and augmented datasets, and reporting evaluation artifacts (confusion matrices, summary CSV, plots). The pipeline also includes a simple knowledge distillation workflow where a teacher MobileNetV2 (trained on the augmented set) teaches a ResNet50 student.

This project is designed for experimentation and reproduction of augmentation effects rather than production deployment.

---

## Key features

* Loads and preprocesses real COVID‑19 and Normal X‑ray images.
* Lightweight CycleGAN implementation (generator + discriminator architectures) to synthesize COVID‑like images from Normal images.
* Automatic generation and saving of synthetic images to `cyclegan_synthetic` and a user-specified `IMAGES_SAVE_PATH`.
* Three classifier architectures for comparison:

  * MobileNetV2 (transfer learning, frozen backbone)
  * Simple custom CNN (from scratch)
  * ResNet50 (transfer learning, frozen backbone)
* Knowledge distillation (teacher → student) implemented in a `Distiller` class.
* Evaluation utilities:

  * Confusion matrices (saved as PNG)
  * Cosine similarity between real and synthetic image features (MobileNetV2 features)
  * Bar chart summary of reported model accuracies and a CSV (`427projectscores.csv`) with results
* Reproducible seeding for NumPy and TensorFlow.

---

## Repository structure (expected)

```
README.md                # This file
main.py (or script)      # The pipeline script (your provided file)
cyclegan_synthetic/      # Generated synthetic images
imagess/                 # Secondary save path used in the script (IMAGES_SAVE_PATH)
confusion_*.png          # Confusion matrix images
427projectscores.csv     # CSV summary of model accuracies
427projectscores.png     # Bar chart of model performance
```

---

## Requirements

* Python 3.8+ recommended
* TensorFlow 2.x (tested with TF 2.8+; GPU recommended for GAN training)
* scikit-learn
* pandas
* matplotlib
* seaborn

Install via pip:

```bash
pip install -r requirements.txt
# or
pip install tensorflow pandas scikit-learn matplotlib seaborn
```

> Note: depending on your GPU/driver setup, you may prefer `tensorflow-cpu` or a specific GPU-enabled TensorFlow build.

---

## Configuration

Before running, set the following top-of-file constants if needed:

* `BASE_PATH_REAL` — path to the COVID-19 Radiography Dataset root (script currently expects `COVID/images` and `Normal/images`).
* `AUG_FOLDER` — output folder where CycleGAN synthetic images are saved (default: `cyclegan_synthetic`).
* `IMAGES_SAVE_PATH` — secondary save location for generated images.
* `NUM_SYNTH_TO_GENERATE` — how many synthetic images to create.
* `IMG_SIZE`, `BATCH_SIZE`, `EPOCHS` — training hyperparameters.

---

## How to run

1. Place the dataset in `BASE_PATH_REAL` (dataset layout: `BASE_PATH_REAL/COVID/images/` and `BASE_PATH_REAL/Normal/images/`).
2. Adjust constants at the top of the file as needed (image size, epochs, paths).
3. Run the script:

```bash
python main.py
```

The pipeline will:

* Load real images
* Train a lightweight CycleGAN (if enough data)
* Generate synthetic images
* Train classifiers on real and augmented datasets
* Run knowledge distillation
* Save confusion matrices, a CSV summary, and a bar chart of reported accuracies

---

## Important implementation notes & caveats

* **`adjust_accuracy()`**: the script contains an `adjust_accuracy` function that intentionally maps raw accuracies to a randomized reported range for certain model names. This appears to be for demonstration or presentation purposes — be transparent about this behavior when publishing results. If you want to report actual, reproducible metrics, remove or disable this function and rely on the raw evaluation metric returned by `model.evaluate`.

* **Small / synthetic training**: The provided CycleGAN is lightweight and trained with a small number of steps by default. For production-quality synthesized images you will need more data, longer training, hyperparameter tuning, and likely a stronger architecture and loss functions (e.g., adversarial losses with patchGAN, learning rate schedulers).

* **Data leakage risk**: Careful with how the augmented dataset is constructed and split. The current pipeline concatenates real and synthetic images then performs a train/test split; ensure synthetic images derived from test set images are not leaking into training folds.

* **Resource usage**: GAN training and large model transfer learning are GPU-intensive. For local testing you can reduce `NUM_SYNTH_TO_GENERATE`, `EPOCHS`, and model sizes.

* **Evaluation**: Add precision/recall/F1 and ROC-AUC if you need more clinical/robust performance reporting. Consider cross‑validation for more robust estimates.

---

