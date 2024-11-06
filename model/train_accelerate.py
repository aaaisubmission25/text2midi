# from aria.data.midi import MidiDict
# from aria.tokenizer import AbsTokenizer
# aria_tokenizer = AbsTokenizer()
import os 
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
import torch.nn as nn
import torch.optim as optim
from accelerate import DistributedDataParallelKwargs
import datasets
import logging
import numpy as np
import yaml
import math
import time
import transformers
from transformers import SchedulerType, get_scheduler
import wandb
import json
import pickle
import os
import random
import deepspeed
from tqdm import tqdm
import torch
from torch import Tensor, argmax
from evaluate import load as load_metric
import sys
import argparse
import jsonlines
from data_loader_remi import Text2MusicDataset
from transformer_model import Transformer
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__)
# Parse command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str, default=os.path.normpath("configs/config.yaml"),
#                     help="Path to the config file")
# parser = deepspeed.add_config_arguments(parser)
# args = parser.parse_args()
config_file = "../configs/config.yaml"
# Load config file
with open(config_file, 'r') as f: ##args.config
    configs = yaml.safe_load(f)

batch_size = configs['training']['text2midi_model']['batch_size']
learning_rate = configs['training']['text2midi_model']['learning_rate']
epochs = configs['training']['text2midi_model']['epochs']
# Artifact folder
artifact_folder = configs['artifact_folder']
# Load encoder tokenizer json file dictionary
tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
# Load the tokenizer dictionary
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)
# Get the vocab size
vocab_size = len(tokenizer)#+1
print("Vocab size: ", vocab_size)
caption_dataset_path = configs['raw_data']['caption_dataset_path']
print(f'caption_dataset_path: {caption_dataset_path}')
# Load the caption dataset
with jsonlines.open(caption_dataset_path) as reader:
    captions = list(reader)
    # captions = [line for line in reader if line.get('test_set') is False]
print("Number of captions: ", len(captions))

def calculate_probability(epoch, epsilon, k, c):
    return max(epsilon, k - c * epoch)

def generate_mixed_input(tgt, predictions, probability):
    batch_size, seq_length = tgt.size()
    mask = torch.rand(batch_size, seq_length) < probability
    mask[:, 0] = False
    mixed_input = tgt.clone()
    mixed_input[mask] = predictions[mask]
    return mixed_input


def collate_fn(batch):
    """
    Collate function for the DataLoader
    :param batch: The batch
    :return: The collated batch
    """
    input_ids = [item[0].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)    
    attention_mask = [item[1].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = [item[2].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return input_ids, attention_mask, labels


# Create the encoder-decoder model
# Initialize the model
d_model = configs['model']['text2midi_model']['decoder_d_model']  # Model dimension (same as FLAN-T5 encoder output dimension)
nhead = configs['model']['text2midi_model']['decoder_num_heads']     # Number of heads in the multiheadattention models
num_layers = configs['model']['text2midi_model']['decoder_num_layers']  # Number of decoder layers
max_len = configs['model']['text2midi_model']['decoder_max_sequence_length']  # Maximum length of the input sequence
use_moe = configs['model']['text2midi_model']['use_moe'] # Use mixture of experts
num_experts = configs['model']['text2midi_model']['num_experts'] # Number of experts in the mixture of experts
dim_feedforward = configs['model']['text2midi_model']['decoder_intermediate_size'] # Dimension of the feedforward network model
gradient_accumulation_steps = configs['training']['text2midi_model']['gradient_accumulation_steps']
use_scheduler = configs['training']['text2midi_model']['use_scheduler']
scheduled_sampling = configs['training']['text2midi_model']['scheduled_sampling']
if scheduled_sampling:
    epsilon = configs['training']['text2midi_model']['epsilon']
    c = configs['training']['text2midi_model']['c']
    k = configs['training']['text2midi_model']['k']
checkpointing_steps = configs['training']['text2midi_model']['checkpointing_steps']
if use_scheduler:
    lr_scheduler_type = configs['training']['text2midi_model']['lr_scheduler_type']
    num_warmup_steps = configs['training']['text2midi_model']['num_warmup_steps']
max_train_steps = configs['training']['text2midi_model']['max_train_steps']
use_deepspeed = configs['model']['text2midi_model']['use_deepspeed'] # Use deepspeed
use_accelerate = configs['model']['text2midi_model']['use_accelerate']
if use_accelerate:
    with_tracking = configs['training']['text2midi_model']['with_tracking']
    report_to = configs['training']['text2midi_model']['report_to']
    output_dir = configs['training']['text2midi_model']['output_dir']
    per_device_train_batch_size = configs['training']['text2midi_model']['per_device_train_batch_size']
    save_every = configs['training']['text2midi_model']['save_every']
assert not (use_deepspeed and use_accelerate), "Exactly one of the parameters must be True and the other must be False"
assert use_scheduler == use_accelerate, "if use accelerate, must use scheduler"
if use_deepspeed:
    ds_config = configs['deepspeed_config']['deepspeed_config_path']
    import deepspeed
    from deepspeed.accelerator import get_accelerator
    local_rank = int(os.environ['LOCAL_RANK']) 
    device = (torch.device(get_accelerator().device_name(), local_rank) if (local_rank > -1)
              and get_accelerator().is_available() else torch.device("cpu"))
    deepspeed.init_distributed(dist_backend='nccl')
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
elif use_accelerate:
    accelerator_log_kwargs = {}
    if with_tracking:
        accelerator_log_kwargs["log_with"] = report_to
        accelerator_log_kwargs["logging_dir"] = output_dir
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision='fp16', kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)], **accelerator_log_kwargs)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    datasets.utils.logging.set_verbosity_error()
    if accelerator.is_main_process:
        if output_dir is None or output_dir == "":
            output_dir = "saved/" + str(int(time.time()))
            
            if not os.path.exists("saved"):
                os.makedirs("saved")
                
            os.makedirs(output_dir, exist_ok=True)
            
        elif output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(output_dir, "outputs"), exist_ok=True)
        # with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
        #     f.write(json.dumps(dict(vars(args))) + "\n\n")
        accelerator.project_configuration.automatic_checkpoint_naming = False
        wandb.init(project="Text to Midi")
    accelerator.wait_for_everyone()
    device = accelerator.device
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
# Load the dataset
if use_accelerate:
    with accelerator.main_process_first():
        dataset = Text2MusicDataset(configs, captions, remi_tokenizer=tokenizer, mode="train", shuffle = True)
        print(dataset[0])
        # dataset = Subset(dataset, indices=range(100))

        dataloader = DataLoader(dataset, batch_size=per_device_train_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)  
else:
    dataset = Text2MusicDataset(configs, captions, remi_tokenizer=tokenizer, mode="train", shuffle = True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
print_every = 10
model = Transformer(vocab_size, d_model, nhead, max_len, num_layers, dim_feedforward, use_moe, num_experts, device=device)
model.load_state_dict(torch.load('/root/test/text2midi/output_test_lr_symph_ft/epoch_50/pytorch_model.bin', map_location=device))
# if use_accelerate:
#     model = torch.nn.DataParallel(model)
# Print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")
# Print number of trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")
if not use_deepspeed:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
if use_scheduler and use_accelerate:
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    if max_train_steps == 'None':
        max_train_steps=None
    print(f'max num of training steps: {max_train_steps}, {type(max_train_steps)}')
    if max_train_steps is None:
        max_train_steps = epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        num_warmup_steps = 20000 #int(0.3*max_train_steps)
        print(f'num_warmup_steps: {num_warmup_steps}')
        
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps, #* gradient_accumulation_steps,
        num_training_steps=max_train_steps, # * gradient_accumulation_steps,
    )
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    dataloader = accelerator.prepare(dataloader)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    checkpointing_steps = checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    # if with_tracking:
    #     experiment_config = vars(configs)
    #     # TensorBoard cannot log Enums, need the raw value
    #     experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    #     accelerator.init_trackers("text_to_midi", experiment_config)
    total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
criterion = nn.CrossEntropyLoss()
def train_model_accelerate(model, dataloader, criterion, num_epochs, max_train_steps, optimizer=None, out_dir=None, checkpointing_steps='epoch', with_tracking=False, save_every=5, device='cpu'):
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 50
    model = model.to(device)
    model.train()
    best_loss = np.inf
    for epoch in range(starting_epoch, num_epochs):
        total_loss = 0
        every_n_steps_loss = 0 
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                encoder_input, attention_mask, tgt = batch
                encoder_input = encoder_input.to(device)
                attention_mask = attention_mask.to(device)
                tgt = tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                if use_moe:
                    outputs, aux_loss = model(encoder_input, attention_mask, tgt_input)
                else:
                    outputs = model(encoder_input, attention_mask, tgt_input)
                    # predictions = outputs.argmax(dim=-1)
                    # probability = calculate_probability(epoch, epsilon, k, c)
                    # mixed_input = generate_mixed_input(tgt_input, predictions, probability)
                    # mixed_outputs = model(encoder_input, attention_mask, mixed_input[:, :-1])
                    # outputs = mixed_outputs
                    aux_loss = 0
                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
                loss += aux_loss
                total_loss += loss.detach().float()
                every_n_steps_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                # if completed_steps % print_every == 0:
                progress_bar.set_postfix({"Loss": loss.item()})
                progress_bar.update(1)
                completed_steps += 1
                if accelerator.is_main_process: # and (completed_steps-1)% print_every == 0:
                    result = {}
                    result["epoch"] = epoch+1
                    result["step"] = completed_steps
                    result["train_loss"] = round(total_loss.item()/(gradient_accumulation_steps*completed_steps),4) # round(total_loss.item()/completed_steps-1 , 4)
                    wandb.log(result)
                    # every_n_steps_loss = 0
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if out_dir is not None:
                        output_dir = os.path.join(out_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= max_train_steps:
                break

                # every_n_steps_loss = 0
        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch+1
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item()/len(dataloader), 4)
            # wandb.log(result)
            result_string = "Epoch: {}, Loss Train: {}\n".format(epoch, result["train_loss"])
            accelerator.print(result_string)
            with open("{}/summary.jsonl".format(out_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")
            logger.info(result)
                
        if accelerator.is_main_process:
            if total_loss < best_loss:
                best_loss = total_loss
                save_checkpoint = True
            else:
                save_checkpoint = False

                # if with_tracking:
                #     accelerator.log(result, step=completed_steps)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and checkpointing_steps == "best":
            if save_checkpoint:
                accelerator.save_state("{}/{}".format(out_dir, "best"))
            if (epoch + 1) % save_every == 0:
                logger.info("Saving checkpoint at epoch {}".format(epoch+1))
                accelerator.save_state("{}/{}".format(out_dir, "epoch_" + str(epoch+1)))

        if accelerator.is_main_process and checkpointing_steps == "epoch":
            accelerator.save_state("{}/{}".format(out_dir, "epoch_" + str(epoch+1)))
                
# torch.cuda.empty_cache()
def train_model(model, dataloader, criterion, num_epochs, optimizer=None):   
    if use_deepspeed:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        model, optimizer, _, _ = deepspeed.initialize(model=model, 
                                                      optimizer=optimizer, 
                                                      model_parameters=model.parameters(),
                                                      config=ds_config)
    else:
        model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(total=int(len(dataloader)/batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for step, batch in enumerate(dataloader):
                if use_deepspeed:
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                
                # Get the batch
                encoder_input, attention_mask, tgt = batch
                # print(encoder_input.shape)
                encoder_input = encoder_input.to(device)
                attention_mask = attention_mask.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                if use_moe:
                    outputs, aux_loss = model(encoder_input, attention_mask, tgt_input)
                else:
                    outputs = model(encoder_input, attention_mask, tgt_input)
                    aux_loss = 0

                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
                loss += aux_loss
                if use_deepspeed:
                    model.backward(loss)
                    model.step()
                else:
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                if step % print_every == 0:
                    pbar.set_postfix({"Loss": loss.item()})
                    pbar.update(1)
            
            pbar.set_postfix({"Loss": total_loss / len(dataloader)})
            pbar.update(1)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

# for i in range(10):
#     print(i)
#     for step, batch in enumerate(dataloader):
#         encoder_input, attention_mask, tgt = batch
        
# Train the model
if use_deepspeed:
    train_model(model, dataloader, criterion, num_epochs=epochs)
elif use_accelerate:
    train_model_accelerate(model, dataloader, criterion, num_epochs=epochs, max_train_steps=max_train_steps,
                           optimizer=optimizer, out_dir=output_dir, checkpointing_steps=checkpointing_steps,
                           with_tracking=with_tracking, save_every=save_every, device = device)
else:
    train_model(model, dataloader, criterion, num_epochs=epochs, optimizer=optimizer)

# Save the trained model
torch.save(model.state_dict(), "transformer_decoder_remi_plus.pth")
print("Model saved as transformer_decoder_remi_plus.pth")
