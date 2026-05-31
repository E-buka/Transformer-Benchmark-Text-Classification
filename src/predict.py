import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import evaluate
import json
import time


def test_metrics(y_true, y_pred):

    load_accuracy = evaluate.load("accuracy")
    load_precision = evaluate.load("precision")
    load_recall = evaluate.load("recall")
    load_f1 = evaluate.load("f1")

    accuracy = load_accuracy.compute(predictions=y_pred, references=y_true)["accuracy"]
    precision = load_precision.compute(predictions=y_pred, references=y_true, average="micro")["precision"]
    recall = load_recall.compute(predictions=y_pred, references=y_true, average='micro', zero_division=0)["recall"]
    f1 = load_f1.compute(predictions=y_pred, references=y_true, average='micro')["f1"]
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "cm":cm.tolist()
    }





def predict(test_data, model, tokenizer, device, batch_size=config.BATCH_SIZE, max_len=config.MAX_SEQ_LENGTH):

    start=time.time()
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    labels = list(test_data['labels'])
    text = test_data['text']


    for i in range(0, len(test_data), batch_size):
        batch = text[i:i+batch_size]

        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)
            pred_probs = probs.amax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(pred_probs.cpu().tolist())

    test_results = test_metrics(y_true=labels, y_pred=all_preds)
    predicted_data = {"text":list(text),
                      "labels":labels,
                      "predictions":all_preds,
                      "probability":all_probs}
    stop = time.time()
    test_results["inference_time"] = round(stop - start , 3)
    test_results["num_of_params"] = model.num_parameters()

    # model_params = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_params])
    # test_results["model_total_params"] = params.tolist()

    return test_results, pd.DataFrame.from_dict(predicted_data)


if __name__ == "__main__":

    test_data = load_dataset("ag_news", split="test")
    test_data = test_data.rename_column("label", "labels")

    model_path = config.BEST_MODEL_PATH

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test_results, predicted_data = predict(test_data, model, tokenizer, device)

    with open(f"{config.RESULT_DIR}/test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    predicted_data.to_csv(config.PRED_DF_NAME, index=False)
