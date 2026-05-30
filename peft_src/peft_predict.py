import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import PeftConfig, PeftModel 
from datasets import load_dataset
from src.predict import test_metrics, predict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src import config
import json
import time


def main():

    test_data = load_dataset("ag_news", split="test")
    test_data = test_data.rename_column("label", "labels")

    model_path = config.BEST_MODEL_PATH
    

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    peft_config = PeftConfig.from_pretrained(model_path)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=4,
    )
    
    model = PeftModel.from_pretrained(
        base_model, 
        model_path
    )
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test_results, predicted_data = predict(test_data, model, tokenizer, device)

    with open(f"{config.RESULT_DIR}/peft_test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    predicted_data.to_csv(config.PRED_DF_NAME, index=False)

if __name__ == "__main__":
    main()
