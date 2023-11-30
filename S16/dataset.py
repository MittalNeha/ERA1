import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        return {
            "encoder_input": enc_input_tokens,
            "decoder_input": dec_input_tokens,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_str_length": len(enc_input_tokens),
            "decoder_str_length": len(dec_input_tokens)
        }

    def collate_batch(self, batch):
        encoder_input_max = max(x["encoder_str_length"] for x in batch)
        decoder_input_max = max(x["decoder_str_length"] for x in batch)
        encoder_input_max += 2
        decoder_input_max += 1 # for the eos token

        encoder_inputs = []
        decoder_inputs = []
        encoder_mask = []
        decoder_mask = []
        labels = []
        src_text = []
        tgt_text = []

        for b in batch:
            enc_input_tokens = b["encoder_input"]  # Includes sos and eos
            dec_input_tokens = b["decoder_input"]

            # Add sos, eos, padding to each sentence
            enc_num_padding_tokens = encoder_input_max - len(enc_input_tokens) - 2
            dec_num_padding_tokens = decoder_input_max - len(dec_input_tokens) - 1

            # Check that number of tokens is positive
            if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
                raise ValueError("Sentence is too short")

            # Add sos and eos token
            encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )

            # Add only sos token for decoder
            decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )

            # only eos token for decoder output
            label = torch.cat(
                [
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )
            # Double-check the size of tensors to make sure they are all seq_len long
            assert encoder_input.size(0) == encoder_input_max
            assert decoder_input.size(0) == decoder_input_max
            assert label.size(0) == decoder_input_max

            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            encoder_mask.append(((encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()).unsqueeze(0)) # (1, 1, seq_len)
            decoder_mask.append(((decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))).unsqueeze(0))
            labels.append(label)
            src_text.append(b["src_text"])
            tgt_text.append(b["tgt_text"])
        return {
            "encoder_input": torch.vstack(encoder_inputs),
            "decoder_input": torch.vstack(decoder_inputs),
            "encoder_mask": torch.vstack(encoder_mask),
            "decoder_mask": torch.vstack(decoder_mask),
            "label": torch.vstack(labels),
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
