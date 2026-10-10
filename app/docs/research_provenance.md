# Research paper provenance

This document records how the five supplied papers are presented in RespiraLab. They are used as related-work context and to explain method families. They are not treated as evidence that the project reproduced a paper's dataset, experiment, accuracy, or hardware implementation. Project-specific settings and results come from this repository's notebooks and saved artifacts.

## Paper-to-project map

| Paper | Relevant project area | What is related | What this project does not claim |
| --- | --- | --- | --- |
| Sabry et al., “Lung Disease Recognition Methods Using Audio-based Analysis with Machine Learning,” *Heliyon*, 2024. [DOI](https://doi.org/10.1016/j.heliyon.2024.e26218) | Notebooks 04–09; Audio & features | Broad survey of respiratory-audio datasets, signal processing, feature extraction, and learning approaches. | The review is not the source of the project's exact 4 kHz target, feature settings, or reported results. |
| Wanasinghe et al., “Lung Sound Classification With Multi-Feature Integration Utilizing Lightweight CNN Model,” *IEEE Access*, 2024. [DOI](https://doi.org/10.1109/ACCESS.2024.3361943) | Notebooks 10–12 and 15; Model development and XAI | Related feature families include Mel, MFCC, and chroma, with lightweight CNN and explainability context. | The paper's data, architecture/configuration, and reported accuracy are not project results. |
| Perera and Pathmakumara, “Advanced Deep Learning Techniques for Lung Sound Classification: Binary, Multi-Class and Ensemble Approach,” CSECS, 2025. [DOI](https://doi.org/10.1109/CSECS64665.2025.11009363) | Notebooks 11–13; Model development | CNN and CNN-LSTM are related model families; the paper also supplies ensemble context. | This app evaluates its saved individual model artifacts. It does not implement that paper's CNN/CNN-LSTM ensemble or claim its scores. |
| Han et al., “Hierarchical Embedded System Based on FPGA for Classification of Respiratory Diseases,” *IEEE Access*, 2025. [DOI](https://doi.org/10.1109/ACCESS.2025.3573162) | Notebooks 05–09; Audio & features and model development | Related respiratory-cycle feature engineering and classification context. | The project is not the paper's hierarchical FPGA system; hardware, sampling setup, cohort, protocol, and reported metrics differ. |
| Nguyen et al., “Lung-Sound Respiratory Disease Classification via Multiple-Instance Learning,” *IEEE Access*, 2026. [DOI](https://doi.org/10.1109/ACCESS.2026.3669914) | Notebooks 14 and 16; Patient analysis and future work | Patient-level learning provides useful context for aggregating multiple respiratory cycles. | This project aggregates saved cycle predictions and does not implement the paper's multi-channel MIL architecture or dataset. |

## Project workflow and evidence

- **Notebooks 01–03:** dataset structure, cleaning and EDA. Dataset-level diagnosis counts are kept distinct from cycle-level respiratory-sound labels.
- **Notebooks 04–08:** audio preprocessing, cycle features, feature analysis/selection, PCA and clustering. The saved project artifacts state a 4 kHz processing target, 39 engineered features, a 33-feature classical-model selection and the recorded PCA/clustering results.
- **Notebooks 09–13:** classical models and neural experiments. Notebook 12 and its 12B regularized variant are two experiments in the multi-feature CNN family; they are not two unrelated model families. Notebook 13 records the CNN-LSTM experiment.
- **Notebooks 14–16:** patient-level aggregation, Grad-CAM examples and model evaluation. Saved patient aggregation is explicitly distinguished from cycle-level evaluation. The app labels saved Grad-CAM figures as research examples; uploaded-audio explanations, when produced, are clearly scoped to the live prediction.
- **Notebooks 17–18:** saved artifact inventory, package/architecture checks and prediction-validation records. The app displays these saved records without loading model weights on informational pages.

## Citation and interpretation policy

The frontend links to DOI landing pages and describes the relevance and difference for each paper. It does not reproduce publisher figures. Do not copy paper-reported accuracy into project metric displays; show the project's own saved metrics, evaluation unit, split and sample support. When an implementation relationship cannot be established from a notebook or saved source record, describe it as related work rather than claiming direct adoption.
