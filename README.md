# App Review Classification for Requirements Engineering

This repository serves as the base for my final year project at the University of Manchester, where I researched the classification of app reviews for requirements engineering using deep learning models: BERT, RoBERTa, and DistilBERT.

## Introduction

App reviews play a crucial role in understanding user requirements and preferences for app development. In this project, we focused on the classification of app reviews to support requirements engineering processes. By leveraging state-of-the-art deep learning models such as BERT, RoBERTa, and DistilBERT, we aimed to automatically extract relevant information from app reviews, enabling developers to gain insights into user expectations and improve the software development lifecycle.

## Table of Contents
- [Dataset](#dataset)
- [Models](#models)
- [Experiments](#experiments)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset used in this project was obtained from two sources: the 'Pan Dataset' and the 'Maalej Dataset'. The Pan Dataset includes reviews from various apps, such as AngryBirds, Dropbox, Evernote, TripAdvisor, PicsArt, Pinterest, and WhatsApp, totaling 1,390 reviews. These reviews were classified into four classes: Feature Request (FR), Problem Discovery (PD), Information Giving (IG), and Information Seeking (IS).

The Maalej Dataset consists of 3,691 reviews from different Google Play and Apple App Store apps. These reviews were classified into four classes: Feature Request (FR), Problem Discovery (PD), User Experience (UE), and Rating (RT).

To create a diverse dataset for training and evaluation, the two datasets were combined, resulting in a total of 4,980 reviews. The classes were made consistent between the datasets, and the "Information Seeking" class from the Pan Dataset was removed as it was not relevant to software engineering requirements.

The combined dataset contains five classes related to software engineering requirements: Problem Discovery (PD), Rating (RT), Feature Request (FR), User Experience (UE), and Information Giving (IG). Undersampling was performed to address class imbalance, ensuring an equal representation of each class.

The final dataset used for training and evaluation consists of 3,970 reviews. The original datasets have been provided along with the combined dataset which was used for the project.

## Models

The project utilizes the following deep learning models for app review classification:

- BERT: Bidirectional Encoder Representations from Transformers
- RoBERTa: A Robustly Optimized BERT Pretraining Approach
- DistilBERT: Distill the Knowledge from BERT

Each model is implemented and provided with relevant training and evaluation scripts.

## Experiments

Two alternative techniques were applied to train and evaluate the deep learning models in this research project:

### Approach 1: Train-Test Split

In the first approach, the combined dataset was randomly split into a training set (75%) and a testing set (25%). The deep learning models were trained on the training set using the Adam optimizer with a learning rate of 2e-5 for 10 epochs. A batch size of 8 was used, and a binary cross-entropy loss function was employed. To prevent overfitting, early stopping was implemented if the validation loss did not improve after 2 epochs.

### Approach 2: 10-Fold Cross Validation

For the second approach, 10-fold cross-validation was utilized to train and evaluate the models. The combined dataset was divided into 10 folds, ensuring each fold contained an equal proportion of samples from each class. The models were trained on each of the 10 folds using the same hyperparameters as in Approach 1. Performance measures such as accuracy, AUROC score, precision, recall, and F1-score were computed for each fold and averaged across all folds.

To monitor the model's development and prevent overfitting, the performance on the validation set was used during training. Early stopping was also employed to halt training when the model converged on the training set.

## Results

The performance of three popular pre-trained language models (BERT, RoBERTa, and DistilBERT) was evaluated on five different software engineering tasks using two different experimental approaches.

### Approach 1: 75%-25% Train-Test Split

- BERT achieved the highest accuracy (87%) and F1-scores, as well as the highest AUROC scores on several tasks.
- RoBERTa and DistilBERT had lower scores overall, but their performance varied across different tasks.
- The differences in architecture, training techniques, and the nature of tasks contributed to the variations in model performance.

### Approach 2: 10-Fold Cross Validation

- All three models showed significant improvement in performance when evaluated using 10-fold cross-validation.
- The average F1-scores for each task and model increased, indicating better generalization capability.
- Cross-validation allowed for training on a larger amount of data, reducing the risk of overfitting and improving performance.
- The UE task resulted in lower scores across all metrics, suggesting its complexity and the need for a nuanced understanding of user feedback.

### Comparison of Models

- DistilBERT consistently underperformed compared to BERT and RoBERTa, possibly due to its smaller size and reduced complexity.
- The smaller size of DistilBERT may have limited its ability to capture intricate connections between words and phrases, especially for more difficult tasks.
- These results emphasize the importance of evaluating model performance on multiple tasks and using different evaluation approaches.

Overall, this research highlights the need for continued improvement and research in natural language processing models to better address complex software engineering tasks.

## Acknowledgments

I would like to give a special acknoledgement to my supervisor Liping Zhao who supported me throughout the entire project.
