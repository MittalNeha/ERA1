# Task
1. Pick the "en-fr" dataset from opus_books <br>
2. Remove all English sentences with more than 150 "tokens" <br>
3. Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10<br>
4. Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8 <br>

## Experiments done
1. d_model=128, len(eng) < 150
2. d_model=128, len(eng) < 150, OneCyclePolicy(pct_start=0.1): <b>Model size: 1.1GB </b>
3. d_model=512, len(eng) < 150, OneCyclePolicy(pct_start=0.2), Dynamic Padding, d_ff = 128: <b>Model size: 800 MB</b>


    def collate_batch(self, batch):
        encoder_input_max = max(x["encoder_str_length"] for x in batch)
        decoder_input_max = max(x["decoder_str_length"] for x in batch)
        encoder_input_max += 2
        decoder_input_max += 1 # for the eos token

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
<i>Loss Trend</i> <br>
Processing Epoch 05: 100%|██████████| 6146/6146 [09:01<00:00, 11.34it/s, loss=2.926]

Processing Epoch 10: 100%|██████████| 6146/6146 [08:54<00:00, 11.49it/s, loss=2.359]

Processing Epoch 15: 100%|██████████| 6146/6146 [09:26<00:00, 10.85it/s, loss=2.018]

Processing Epoch 20: 100%|██████████| 6146/6146 [08:57<00:00, 11.43it/s, loss=1.728]
<br><br> <i> Comments: </i>
- Loss Trend looks good. 
- Model size has reduced


4. d_model=512, len(eng) < 150, OneCyclePolicy(, Dynamic Padding, d_ff = 128, Parameter sharing:<b>Model size: 676.9</b>

<br><i>Main Changes </i><br>


    def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
        ...
        e1, e2, e3 = encoder_blocks
        d1, d2, d3 = decoder_blocks
    
        encoder_blocks_ps = [e1, e2, e3, e1, e2, e3]
        decoder_blocks_ps = [d1, d2, d3, d1, d2, d3]
    
        encoder = Encoder(nn.ModuleList(encoder_blocks_ps))
        decoder = Decoder(nn.ModuleList(decoder_blocks_ps))