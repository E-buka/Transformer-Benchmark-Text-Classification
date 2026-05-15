from peft import LoraConfig, TaskType, get_peft_model 
import utils 
import config
import json 
from train import is_complete_checkpoint, find_latest_checkpoint, tokenizer, tokenization
from seed import set_seed
from load_data import prepare_dataset
from log_builder import build_logger
from transformers import Trainer, EarlyStoppingCallback, DataCollatorWithPadding
import time



def main():
    set_seed()
    
    logger, file_handler, console_handler = build_logger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
   
    logger.debug("Loading train and validation datasets for PEFT model training")
    
    train_data, eval_data = prepare_dataset() 

    data_collator = DataCollatorWithPadding(tokenizer)
    train_data = train_data.map(tokenization, batched=True)
    eval_data = eval_data.map(tokenization, batched=True)
    
    if not train_data and eval_data:
        logger.critical(f"Failure preparing data!!!")

    logger.debug("Setting LORA Configurations, base model")
    
    lora_config = LoraConfig(
        task_type = TaskType.SEQ_CLS, 
        inference_mode=False,  
        r= config.LORA_R,
        lora_alpha = config.LORA_ALPHA,
        lora_dropout= config.LORA_DROPOUT,
    )
    
    model = utils.model(config.MODEL_NAME, config.NUM_LABELS)

    lora_model = get_peft_model(model, lora_config)
    lora_model_parameters = lora_model.print_trainable_parameters()

    with open(config.RESULT_DIR/"peft_params.json", "w") as p:
        json.dump(lora_model_parameters, p, indent=2)

    training_args = utils.build_train_args ()
    training_args.output_dir = f"{config.OUTPUT_DIR}/peft"
    training_args.logging_dir = f"{config.LOG_DIR}/peft"  
   

    trainer = Trainer(
            model = lora_model, 
            args = training_args, 
            train_dataset= train_data, 
            eval_dataset = eval_data,
            processing_class = tokenizer,
            data_collator = data_collator,
            compute_metrics = utils.compute_metrics,
            preprocess_logits_for_metrics=utils.preprocess_logits_for_metrics, #keep eye
            callbacks = [EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOP_PATIENCE,
                                            early_stopping_threshold= config.EARLY_STOP_THRESHOLD),]
            
        )
    
    logger.debug(f"Model selected as >>> {config.MODEL_NAME} <<< ") 
    logger.info(f"Model parameters: {lora_model_parameters}")
    logger.info("============BATCH TRAINING INFORMATION============ "
        f"Train size: {len(train_data)} "
        f"Batch size: {config.BATCH_SIZE} "
        f"Epochs: {config.EPOCHS} "
        f"Steps per epoch: {len(train_data) // config.BATCH_SIZE} "
        f"Approx total steps: {(len(train_data) // config.BATCH_SIZE) * config.EPOCHS}"
    )
    
    
    last_checkpoint = find_latest_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        logger.info (f"Resuming training from checkpoint: {last_checkpoint}")
    else: 
        logger.info("No checkpoint found. Starting fresh training run.")
        
        
    start = time.time() 
    trainer.train(resume_from_checkpoint=last_checkpoint) 
    trainer.evaluate()   
    stop = time.time() 
    train_time = {"Training_time": round(stop-start, 3)}
    
    with open(config.RESULT_DIR/"peft_train_time.json") as f:
        json.dump(train_time, f, indent=2)
    
    best_model_path = f"{training_args.output_dir}/best_model" 
    lora_model.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    logger.info("PEFT Model training completed successfully with best model and tokenizer saved")
    logger.removeHandler(console_handler)
    logger.removeHandler(file_handler)
    
if __name__ == "__main__":
    main()
    