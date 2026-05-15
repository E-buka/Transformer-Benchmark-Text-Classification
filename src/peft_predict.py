from peft import AutoPeftModel
from datasets import load_dataset 
from predict import test_metrics, predict 
from transformers import AutoTokenizer
import torch
import config
import json
import time


def main():
    
    test_data = load_dataset("ag_news", split="test") 
    test_data = test_data.rename_column("label", "labels")  
     
    model_path = config.BEST_MODEL_PATH
    
    model = AutoPeftModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test_results, predicted_data = predict(test_data, model, tokenizer, device)
    
    with open(config.RESULT_DIR/"peft_test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    predicted_data.to_csv(config.PRED_DF_NAME, index=False)
    
if __name__ == "__main__":
    main()