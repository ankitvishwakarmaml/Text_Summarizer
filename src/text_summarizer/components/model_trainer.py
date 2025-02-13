
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset, load_from_disk
from text_summarizer.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model_T5 = T5ForConditionalGeneration.from_pretrained(self.config.model_name).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_T5)
        
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = Seq2SeqTrainingArguments(
             output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, 
             warmup_steps=self.config.warmup_steps, per_device_train_batch_size=self.config.per_device_train_batch_size, 
             per_device_eval_batch_size=self.config.per_device_train_batch_size, weight_decay=self.config.weight_decay, 
             logging_steps=self.config.logging_steps, evaluation_strategy=self.config.evaluation_strategy,
             eval_steps=self.config.eval_steps, save_steps=self.config.save_steps, logging_dir=self.config.logging_dir,
             learning_rate=self.config.learning_rate, gradient_accumulation_steps=self.config.gradient_accumulation_steps,
             fp16=True, predict_with_generate=True
         ) 


        #trainer_args = Seq2SeqTrainingArguments(
        #    output_dir=self.config.root_dir, evaluation_strategy="steps", eval_steps=500, save_steps=500,
        #    learning_rate=3e-5, per_device_train_batch_size=8, per_device_eval_batch_size=8, num_train_epochs=3,
        #    weight_decay=0.01, warmup_steps=500,
        #    fp16=True,  # Enable mixed precision training
        #    predict_with_generate=True # Generate summaries during evaluation
        #) 

        trainer = Seq2SeqTrainer(model=model_T5, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["train"], 
                  eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        ## Save model
        model_T5.save_pretrained(os.path.join(self.config.root_dir,"T5-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
