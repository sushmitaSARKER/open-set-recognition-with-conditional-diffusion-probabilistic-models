# open-set-recognition-with-conditional-diffusion-probabilistic-models

This repository contains the PyTorch implementation of a model for open-set recognition (OSR) of radio frequency (RF) signals. The methodology is based on the principles outlined in the paper **"RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion"**. The core idea is to train a conditional diffusion model to reconstruct signals from known classes. The reconstruction error is then used as an anomaly score to distinguish between known and unknown signals.

The project is structured into a three-phase workflow:
1.  **Feature Extractor Training:** A disentangled feature extractor is trained to generate conditioning vectors for the diffusion model.
2.  **Conditional Diffusion Model Training:** A Hierarchical Diffusion Transformer (`tfdiff_WiFi`) is trained to reconstruct signals of known classes, guided by the conditioning vectors.
3.  **Evaluation:** The trained models are used to perform open-set recognition on a test set by calculating reconstruction errors and applying an optimal threshold.

---

## File Structure

The repository is organized as follows:

.├── models/│   ├── init.py│   ├── feature_extractor.py  # Defines the DisentangledFeatureExtractor model│   ├── diffusion_model.py    # Defines the tfdiff_WiFi (diffusion transformer) model│   └── complex_layers.py     # Contains custom complex-valued PyTorch layers├── utils/│   ├── init.py│   ├── diffusion_helper.py   # Implements the diffusion forward process│   └── loss_functions.py     # Defines the CosineSimilarityLoss├── data/│   └── ... (Your .mat dataset files should go here)├── saved_models/│   └── ... (Trained models will be saved here)├── config.py                 # Main configuration file for paths, and model parameters├── data_loader.py            # Handles loading and preprocessing of the .mat dataset files├── train.py                  # Main script for running the training phases├── engine.py                 # Contains the core training and evaluation loops└── evaluate.py               # Script for running the final OSR evaluation
---

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch
- NumPy
- SciPy (for loading `.mat` files)
- scikit-learn (for evaluation metrics)
- tqdm (for progress bars)
- Matplotlib (for plotting, optional)

You can install them using pip:
```bash
pip install torch numpy scipy scikit-learn tqdm matplotlib
Getting StartedFollow these steps to configure, train, and evaluate the model.1. Dataset PreparationThis project expects the dataset to be in .mat format.Place your .mat files in a directory (e.g., data/).The data loader (data_loader.py) assumes each .mat file contains a variable named vect where the signal data is stored.2. ConfigurationAll important parameters and file paths must be set in config.py. Open this file and modify the following sections:PATHS: Set the paths for your dataset files and the directories where you want to save the trained models.FEATURE_EXTRACTOR_PARAMS: Configure parameters for the feature extractor, such as the number of known classes and feature dimensions.DIFFUSION_PARAMS: Configure parameters for the diffusion model, such as the number of diffusion steps (max_step).TRAINING_PARAMS: Set training hyperparameters like learning rates, batch size, and the number of epochs for each phase.3. Training the ModelsThe training process is split into two phases. You can run them sequentially or one at a time using the train.py script.To run both phases sequentially:python train.py
To run only Phase 1 (Feature Extractor Training):python train.py --phase 1
This will train the DisentangledFeatureExtractor and save the model weights to the path specified in config.py.To run only Phase 2 (Diffusion Model Training):Make sure you have already completed Phase 1, as this step requires the trained feature extractor model.python train.py --phase 2
This will load the feature extractor, train the tfdiff_WiFi diffusion model, and save its weights.4. EvaluationAfter both models are trained, you can run the final open-set recognition evaluation on your test set.The evaluate.py script will:Load the trained feature extractor and diffusion models.Use the threshold_loader data to calculate an optimal reconstruction error threshold using Youden's Index.Use the test_loader data to perform the final OSR classification.Print a detailed report including overall accuracy, F1-score, precision, recall, and a confusion matrix.To run the evaluation:python evaluate.py
Methodology OverviewPhase 1: Disentangled Feature ExtractorGoal: To learn a mapping from a raw RF signal to a robust, low-dimensional feature vector c.Architecture: A two-branch CNN that processes the time and frequency domains of the signal in parallel using complex-valued layers.Loss Function: A composite loss that combines Cross-Entropy (for classification accuracy) and Cosine Similarity (to encourage disentanglement between the time and frequency features).Phase 2: Conditional Diffusion ModelGoal: To train a generative model capable of reconstructing signals from the known classes, conditioned on the feature vector c.Architecture: tfdiff_WiFi, a Hierarchical Diffusion Transformer that uses attention mechanisms and complex-valued layers. It takes a noisy signal, a timestep t, and the condition c as input and predicts the original, clean signal.Training: The model is trained to minimize the Mean Squared Error (MSE) between its reconstruction and the original signal.Phase 3: Open-Set RecognitionAnomaly Score: The reconstruction error (MSE) from the diffusion model is used as an anomaly score. The intuition is that the model will produce low error for known signals it was trained on and high error for unknown signals.Thresholding: An optimal threshold is calculated from a validation set to create a decision boundary.Classification:If error < threshold, the signal is "known" and is passed to the feature extractor for specific class identification.If error >= threshold, the signal is "unknown".CitationThis code is an implementation of the concepts described in the following paper. If you use this code in your research, please consider citing:@article{chi2024rfdiffusion,
  title={RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion},
  author={Chi, Guoxuan and Yang, Zheng and Wu, Chenshu and Xu, Jingao and Gao, Yuchong and Liu, Yunhao and Han, Tony Xiao},
  journal={arXiv preprint arXiv:2404.09140},
  year={2024}
}
