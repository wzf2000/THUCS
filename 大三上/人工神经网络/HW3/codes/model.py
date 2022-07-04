import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rnn_cell import RNNCell, GRUCell, LSTMCell

class RNN(nn.Module):
    def __init__(self,
            num_embed_units,  # pretrained wordvec size
            num_units,        # RNN units size
            num_vocabs,       # vocabulary size
            wordvec,            # pretrained wordvec matrix
            dataloader):      # dataloader

        super().__init__()

        # load pretrained wordvec
        self.wordvec = wordvec
        # the dataloader
        self.dataloader = dataloader

        # TODO START
        # fill the initialization of cells
        self.cell = RNNCell(num_embed_units, num_units)
        # self.cell = GRUCell(num_embed_units, num_units)
        # self.cell = LSTMCell(num_embed_units, num_units)
        # TODO END

        # intialize other layers
        self.linear = nn.Linear(num_units, num_vocabs)

    def forward(self, batched_data, device):
        # Padded Sentences
        sent = torch.tensor(batched_data["sent"], dtype=torch.long, device=device) # shape: (batch_size, length)
        # An example:
        #   [
        #   [2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
        #   [2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
        #   [2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
        #   ]
        # You can use self.dataloader.convert_ids_to_sentence(sent[0]) to translate the first sentence to string in this batch.

        # Sentence Lengths
        length = torch.tensor(batched_data["sent_length"], dtype=torch.long, device=device) # shape: (batch)
        # An example (corresponding to the above 3 sentences):
        #   [5, 3, 6]

        batch_size, seqlen = sent.shape

        # TODO START
        # implement embedding layer
        embeddings = nn.Embedding.from_pretrained(self.wordvec)
        embedding = embeddings(sent) # shape: (batch_size, length, num_embed_units)
        # TODO END

        now_state = self.cell.init(batch_size, device)

        loss = 0
        logits_per_step = []
        for i in range(seqlen - 1):
            hidden = embedding[:, i]
            hidden, now_state = self.cell(hidden, now_state) # shape: (batch_size, num_units)
            logits = self.linear(hidden) # shape: (batch_size, num_vocabs)
            logits_per_step.append(logits)

        # TODO START
        # calculate loss
        len_range = torch.arange(seqlen - 1, device=device)
        valid_mask = len_range[None, :] < length[:, None]
        total = valid_mask.float().sum()
        logits_stack = torch.stack(logits_per_step, dim=-1) # shape: (batch_size, num_vocabs, length - 1)
        loss = F.cross_entropy(logits_stack, sent[:, 1:], reduction='none')
        loss = (loss * valid_mask).sum() / total
        # TODO END

        return loss, torch.stack(logits_per_step, dim=1)

    def inference(self, batch_size, device, decode_strategy, temperature, max_probability):
        # First Tokens is <go>
        now_token = torch.tensor([self.dataloader.go_id] * batch_size, dtype=torch.long, device=device)
        flag = torch.tensor([1] * batch_size, dtype=torch.float, device=device)

        now_state = self.cell.init(batch_size, device)

        generated_tokens = []
        for _ in range(50): # max sentecne length

            # TODO START
            # translate now_token to embedding
            embeddings = nn.Embedding.from_pretrained(self.wordvec)
            embedding = embeddings(now_token) # shape: (batch_size, num_embed_units)
            # TODO END

            hidden = embedding
            hidden, now_state = self.cell(hidden, now_state)
            logits = self.linear(hidden) # shape: (batch_size, num_vocabs)

            if decode_strategy == "random":
                prob = (logits / temperature).softmax(dim=-1) # shape: (batch_size, num_vocabs)
                now_token = torch.multinomial(prob, 1)[:, 0] # shape: (batch_size)
            elif decode_strategy == "top-p":
                # TODO START
                # implement top-p samplings
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits / temperature, dim=-1), dim=-1)
                sorted_delete = cumsum_probs > max_probability
                sorted_delete[:, 1:] = sorted_delete[:, :-1].clone()
                sorted_delete[:, 0] = 0
                delete = sorted_delete.scatter(1, sorted_indices, sorted_delete)
                logits[delete] = -float("inf")
                prob = (logits / temperature).softmax(dim=-1) # shape: (batch_size, num_vocabs)
                now_token = torch.multinomial(prob, 1)[:, 0]  # shape: (batch_size)
                # TODO END
            else:
                raise NotImplementedError("unknown decode strategy")

            generated_tokens.append(now_token)
            flag = flag * (now_token != self.dataloader.eos_id)

            if flag.sum().tolist() == 0: # all sequences has generated the <eos> token
                break

        return torch.stack(generated_tokens, dim=1).detach().cpu().numpy()
