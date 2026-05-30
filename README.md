# Transformer Benchmark Text Classification

## Overview
This end-to-end ML project benchmarks Hugging Face pretrained transformer models for NLP text classification. The project is divided into three subsections. 
1. The first part compares the performance and tradeoffs for two full fine-tuned transformer models. The models are 
 - BERT
 - DistilBERT 
2. The second part selects the best model from the first section and compares the tradeoffs between full fine-tuning and parameter efficient fine-tuning with LORA
3. The final part serves the recommended model using an FastAPI and cloud hosting. 

Dataset:  Hugging Face AG_news dataset

Both models were trained and tested on the same train and test dataset using a consistent preprocessing pipeline, and then compared by the same performance metrics, error analysis, training and inference time. Both models achieved similar performance but differed in training and testing time.  DistilBERT achieved the overal best performance and was selected for PEFT comparison: 

DistilBERT performance result:
- **93%** overall performance for accuracy, precision, recall and F1
- **Training time**: 7hrs, 8min
- **Test inference time**: 4min, 21sec
- **Model size**: 256MB

An Efficiency comparison was done by full fine-tuning and parameter-efficient fine-tuning of the DistilBERT transformer model. The resources cost was significantly reduced for the model by PEFT while achieving similar performance as against full fine-tuning of the model.  

PEFT performance: 
- **93%** overall performance for accuracy, precision, recall and F1
- **Training time**: 1hrs, 44min
- **Test inference time**: 22sec
- **Number of trainable parameters**: 1.1% (741,124 of 67,697,672)
- **Model size**: 3.51MB

The PEFT model was served using FastAPI app for inference on the cloud and can be accessed using the link to the server: 

Link:

The project also includes: 
- hyperparameter search with BERT transformer model using a subset of the train dataset. (Only a subset of the dataset and BERT model was used due to limited computing resources)
- notebook-based error analysis 

## Folder Structure

``` text
transformer-model/
├── data/
├── logs/
│   ├── peft/
│   ├── transformer-models/
├── notebook/
├── outputs/
│   ├── distilbert/
│   ├── peft/
│   └── transformer-models/
├── parameter_tuning/
│   ├── bert/
├── peft_src/
│   ├── __init__.py
│   ├── peft_predict.py
│   ├── peft_train.py
├── results/
│   ├── bert/
│   ├── distilbert/
│   ├── peft/distilbert/
│   ├── transformer_models
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── load_data.py
│   ├── log_builder.py
│   ├── parameter_search.py
│   ├── predict.py
│   ├── seed.py
│   ├── train.py
│   └── utils.py
├── data_split_result.json
├── README.md
└── requirements.txt
``` 
## How to Run
```bash
pip install -r requirements.txt
``` 

### Training
Model to train is selected in src/config.py through MODEL_NAME

Two training options are available: 

- **Full fine-tuning**: selected by setting `peft` to False in src/config.py
>> Full tine-tuning script and the rest of the model resources are set in `src/` package
- **PEFT training**: selected by setting `peft` to True in src/config.py 
>> PEFT training and prediction script are set and controlled from `peft_src/` package

Training will: 
- load the selected pretrained model and tokenizer
- load, tokenize and pad the dataset
- fine-tune the model
- resume from a checkpoint if available
- save the training time to results/
- save the best model and tokenizer

```bash
python src/train.py
``` 
```bash
python peft_src/peft_train.py
```

Example saved model path: 

```bash
output/peft/distilbert/best_model/
```

#### Dataset class balance
Dataset class balance can be evaluated from src/load_data.py. The result is save to the root folder. 

```bash
python src/load_data.py
``` 

#### Hyperparameter search
Hyperparameter search is set in src/parameter_search.py and the model to search is set in src/config.py through MODEL_NAME.
The best parameters are saved to results/{model_name}

### Prediction and Inference

#### Prediction
Test data prediction is generated and the test metrics are computed for the test data. The test metrics are saved to results/{model_name} while the predicted test data are saved to data/ 
Two test prediction script are availble: 
- **full fine-tuning test prediction**: src/predict.py
- **peft test prediction**: peft_src/peft_predict.py 

```bash
python src/predict.py
```
```bash
python peft_src/peft_predict.py
```

#### Inference
Loads the PEFT trained model and predicts a class for a single text or lists of texts. 

```bash

```
## API 
FastAPI

## Notebooks
- colab_train.ipynb -- Colab model fine-tuning and test data prediction 
- Peft vs full training comparison.ipynb -- Tradeoffs comparisions
- Transformer full training error analysis.ipynb -- evaluation and error analysis

## Limitations 

## Conclusion
The project is an end-to-end NLP for text classification which benchmarks two transformer models, compares full-tuning vs parameter efficient fine-tuning, evaluation and API serving. BERT and DistilBERT transformer models were compared on the same dataset, full fine-tuning and PEFT tuning of DistilBERT model were compared, with greater efficiency achieved with PEFT fine-tuning. 