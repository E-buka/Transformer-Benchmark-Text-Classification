import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import pipeline 
from src import config  
from peft import PeftConfig, PeftModel 
from transformers import AutoTokenizer, AutoModelForSequenceClassification 


def load_model():
    
    model_path = config.MODEL_TO_SERVE 
    
    id2label = {
        0: "World", 
        1: "Sports", 
        2: "Business", 
        3: "Sci/Tech",
    }
    
    label2id = {label: idx for idx, label in id2label.items()}
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    peft_config = PeftConfig.from_pretrained(model_path)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    model = PeftModel.from_pretrained(
        base_model, 
        model_path
    )  
    model.eval()  
    
    classifier = pipeline(task="text-classification", 
                          model=model,
                          tokenizer=tokenizer)
    
    return classifier

def text_predictor(text: str, pipeline):
    output = pipeline(text) 
    return output  
