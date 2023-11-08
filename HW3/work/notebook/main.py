import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import (
  example_transform, 
  custom_transform
)
import os

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", device=device)

# Tokenize the input
def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", device=device)
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):

    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    ################################
    ##### YOUR CODE BEGINGS HERE ###
    
    # Implement the training loop --- make sure to use the optimizer and lr_sceduler (learning rate scheduler)
    # Remember that pytorch uses gradient accumumlation so you need to use zero_grad (https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)
    # You can use progress_bar.update(1) to see the progress during training
    # You can refer to the pytorch tutorial covered in class for reference
    
    
    raise NotImplementedError


    ##### YOUR CODE ENDS HERE ######        
    
    print("Training completed...")
    print("Saving Model....")
    model.save_pretrained(save_dir)
    
    return
    
    
# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    metric = evaluate.load("accuracy")
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0 
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        
        # write to output file
        for i in range(predictions.shape[0]):
            out_file.write(str(predictions[i].item()) + "\n")
            #out_file.write("\n")
            out_file.write(str(batch["labels"][i].item()) + "\n\n")
            #out_file.write("\n\n")

    score = metric.compute()
    print(f"Eval score: {score}")
    
    return score


# Created a dataladoer for the augmented training dataset
def create_augmented_dataloader(dataset):
    
    ################################
    ##### YOUR CODE BEGINGS HERE ###
    
    # Here, 'dataset' is the original dataset. You should return a dataloader called 'train_dataloader' (with batch size = 8) -- this
    # dataloader will be for the original training split augmented with 5k random transformed examples from the training set.
    # You may want to set load_from_cache_file to False when using dataset maps
    # You may find it helpful to see how the dataloader was created at other place in this code.
    
    train_dataloader = None
    
    raise NotImplementedError
    
    
    ##### YOUR CODE ENDS HERE ######
    
    return train_dataloader


# Create a dataloader for the transformed test set
def create_transformed_dataloader(dataset, debug_transformation):
    
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('='*30)

        exit()
      
    
    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)                                                    
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset    
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=8)
    
    return eval_dataloader   
