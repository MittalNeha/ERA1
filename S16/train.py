import os
from pathlib import Path

import torch
import torch.nn as nn
import torchmetrics
import warnings

# Huggingface datasets and tokenizers
from datasets import load_dataset, load_from_disk


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, get_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1,).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    if writer:
        # Evaluate the character error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
    
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    if config['ds_loc'] == 'disk':
        ds_raw = load_from_disk(config['ds_path'])
    else:
        ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build Tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    sorted_train_ds = sorted(train_ds_raw, key= lambda x:len(x["translation"][config["lang_src"]]))
    filtered_sorted_train_ds = [k for k in sorted_train_ds if len(k["translation"][config["lang_src"]]) < 150]
    # filtered_sorted_train_ds = [k for k in filtered_sorted_train_ds if len(k["translation"][config["lang_tgt"]]) < 120]
    filtered_sorted_train_ds = [k for k in filtered_sorted_train_ds if len(k["translation"][config["lang_src"]]) + 10 > len(k["translation"][config["lang_tgt"]])]

    filtered_val_ds = [k for k in val_ds_raw if len(k["translation"][config["lang_src"]]) < 150]
    filtered_val_ds = [k for k in filtered_val_ds if
                                len(k["translation"][config["lang_src"]]) + 10 > len(
                                    k["translation"][config["lang_tgt"]])]

    train_ds = BilingualDataset(filtered_sorted_train_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(filtered_val_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the max len of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    max_len_src_sent = 0
    max_len_tgt_sent = 0

    for item in filtered_sorted_train_ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src_sent = max(max_len_src_sent, len(item['translation'][config['lang_src']]))
        max_len_tgt_sent = max(max_len_tgt_sent, len(item['translation'][config['lang_tgt']]))
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence token: {max_len_src}")
    print(f"Max length of target sentence token: {max_len_tgt}")

    print(f"Max length of source sentence: {max_len_src_sent}")
    print(f"Max length of target sentence: {max_len_tgt_sent}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=train_ds.collate_batch)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_batch)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def collate_fn(batch):
    encoder_input_max = max(x["encoder_str_length"] for x in batch)
    decoder_input_max = max(x["decoder_str_length"] for x in batch)

    encoder_inputs = []
    decoder_inputs = []
    encoder_mask = []
    decoder_mask = []
    label = []
    src_text = []
    tgt_text = []

    for b in batch:
        enc_input_tokens = b["encoder_input"]  # Includes sos and eos
        dec_input_tokens = b["decoder_input"]

        # Add sos, eos, padding to each sentence
        enc_num_padding_tokens = encoder_input_max - len(enc_input_tokens)
        dec_num_padding_tokens = decoder_input_max - len(dec_input_tokens)

        # Check that number of tokens is positive
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too short")

        # only eos token for decoder output
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        encoder_inputs.append(b["encoder_input"][:encoder_input_max])
        decoder_inputs.append(b["encoder_input"][:decoder_input_max])
        encoder_mask.append((b["encoder_mask"][0, 0, :encoder_input_max]).unsqueeze(0).unsqueeze(0).unsqueeze(0))
        decoder_mask.append((b["decoder_mask"][0, :decoder_input_max, :decoder_input_max]).unsqueeze(0).unsqueeze(0))
        label.append(b["label"][:decoder_input_max])
        src_text.append(b["src_text"])
        tgt_text.append(b["tgt_text"])
    # print(len(encoder_inputs))
    return  {
        "encoder_input": torch.vstack(encoder_inputs),
        "decoder_input": torch.vstack(decoder_inputs),
        "encoder_mask": torch.vstack(encoder_mask),
        "decoder_mask": torch.vstack(decoder_mask),
        "label": torch.vstack(label),
        "src_text": src_text,
        "tgt_text": tgt_text
    }

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], d_model=config["d_model"], d_ff=config["d_ff"])
    return model

def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    STEPS_PER_EPOCH = len(train_dataloader)
    MAX_LR = 10**-3

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=MAX_LR,
                                                    steps_per_epoch=STEPS_PER_EPOCH,
                                                    epochs=config["num_epochs"],
                                                    pct_start=0.2,
                                                    div_factor=10,
                                                    three_phase=True,
                                                    final_div_factor=10,
                                                    anneal_strategy="linear"
                                                    )

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloadng model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print("Preloaded")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output) # (0, seq_len, vocab_size)

            label = batch['label'].to(device)

            # Compute loss using simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            scheduler.step()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    cfg = get_config()
    cfg['batch_size'] = 10
    cfg['preload'] = None
    cfg['num_epochs'] = 30
    train_model(cfg)