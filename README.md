# Transformer Benchmark Text Classification

## Overview
This end-to-end deep learning project benchmarks Hugging Face pretrained transformer models for text classification on the AG News dataset. The project is divided into three subsections. 
1. The first part compares the performance and trade-offs of two fully fine-tuned transformer models. The models are 
 - BERT
 - DistilBERT 
2. The second part selects the best model from the first section and compares the tradeoffs between full fine-tuning and parameter efficient fine-tuning with LORA
3. The final part containerizes the model using Docker and serves an API prediction endpoint using FastAPI with local serving. 

Dataset:  Hugging Face AG_news dataset

Both models were trained on the same 25,000-example training subset and evaluated on the same test set using a consistent preprocessing pipeline, and then compared by the same performance metrics, error analysis, training and inference time. Both models achieved similar performance but differed in training and testing time.  DistilBERT achieved the overall best performance and was selected for PEFT comparison: 

DistilBERT performance result:
- **93%** overall performance for accuracy, precision, recall and F1 (micro averaging)
- **Training time**: 7 hr 8 min
- **Test inference time**: 4 min 21 sec
- **Model size**: 256 MB

An Efficiency comparison was done with full fine-tuning and parameter-efficient fine-tuning of the DistilBERT transformer model using the entire train dataset. The resource cost was significantly reduced with PEFT while achieving similar performance with full fine-tuning.  

PEFT performance: 
- **93%** overall performance for accuracy, precision, recall and F1 (micro averaging)
- **Training time**: 1 hrs 44 min
- **Test inference time**: 22 sec
- **Number of trainable parameters**: 1.1% (741,124 of 67,697,672)
- **Model size**: 3.51 MB
*Both prediction timings were measured on the same GPU device with the same test set and batch strategy.*

The PEFT model was dockerized and served using FastAPI app endpoint for inference and can be accessed locally. 

The project also includes: 
- hyperparameter search with BERT transformer model using a subset of the train dataset. (Only a subset of the dataset and BERT model was used due to limited computing resources)
- notebook-based error analysis and comparative summary

## Folder Structure

``` text
Transformer-Benchmark-Text-Classification/
├── data/
├── logs/
│   ├── peft/
│   ├── transformer_models/
├── notebooks/
├── output/
│   ├── distilbert/
│   ├── peft/
│   └── transformer_models/
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
│   ├── inference.py
│   ├── load_data.py
│   ├── log_builder.py
│   ├── parameter_search.py
│   ├── predict.py
│   ├── seed.py
│   ├── train.py
│   └── utils.py
├── data_split_result.json
├── README.md
├── Dockerfile
└── requirements.txt
``` 
## How to Run
```bash
pip install -r requirements.txt
``` 

### Training
src/config.py controls model selection adn subset behavior. Model to train is selected in src/config.py through MODEL_NAME

Two training options are available: 

- **Full fine-tuning**: src/train.py if for full fine-tuning, selected by setting `peft` to False in src/config.py

- **PEFT training**: peft_src/peft_train.py is for LoRA fine-tuning, selected by setting `peft` to True in src/config.py 



Training will: 

- load the selected pretrained model and tokenizer
- load, tokenize and pad the dataset
- select the train and evaluation data subset if set
- fine-tune the model
- resume from a checkpoint if available
- save the training time to results/
- save the best model and tokenizer to output/

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
Dataset class balance can be evaluated from src/load_data.py. The result is saved to the root folder. 

```bash
python src/load_data.py
``` 

#### Hyperparameter search
Hyperparameter search is set in src/parameter_search.py and the model to search is set in src/config.py through MODEL_NAME.
The best parameters are saved to results/{model_name}

### Prediction and Inference

#### Prediction
Test data prediction is generated and the test metrics are computed for the test data. The test metrics are saved to results/{model_name} while the predicted test data are saved to data/ 

Two test prediction scripts are available: 
- **full fine-tuning test prediction**: src/predict.py
- **peft test prediction**: peft_src/peft_predict.py 

```bash
python src/predict.py
```

```bash
python peft_src/peft_predict.py
```

#### Inference
Loads the PEFT trained model and predicts the class of a text string or lists of text strings. 

```bash
python src/inference.py
```

## API 
The model is containerized using Docker and served with FastAPI. The FastAPI app loads the peft model on startup and predicts through the API endpoint. 

Run locally with uvicorn: 
```bash
uvicorn src.app:app --reload
```

Run locally with Docker: 
```bash
docker compose up -d
```
##### Example request: 

```json
{"text": "Stocks rose after the company reported strong earnings."}
```

##### Example response:
```json
{'label': 'Business', 'score':0.9793215385744965}
```

## Notebook
- Summary error analysis and performance.ipynb -- Tradeoffs and error analysis summary

## Limitations 
- The hyperparameter search was conducted for only a single pretrained model and with a small data subset due to limited resources. The results of the search may not represent the best hyperparameters for the entire dataset training. 
- The first section of the project which compares BERT and DistilBERT transformer models were trained using a subset (25,000) of the entire train dataset due to limited GPU access. 
- The models were only trained for 5 epochs and therefore further training may likely improve the overall model performances.
- The model performances were evaluated using micro averaging which may not fully expose where the model struggles between various class labels.

## Conclusion
This project built an end-to-end text classification workflow using transformer benchmarking, PEFT-based efficiency comparison, and FastAPI deployment. DistilBERT achieved the best trade-off among the fully fine-tuned models, while LoRA-based PEFT retained similar performance with much lower training cost and model size.