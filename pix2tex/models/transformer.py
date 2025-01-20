import torch
import torch.nn.functional as F
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper, top_k, top_p
from x_transformers import TransformerWrapper, Decoder


class BeamHypotheses(object):
    def __init__(self, num_beams: int, length_penalty: float):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int):
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def beam_generate(self, start_tokens, context, seq_len=256,  **kwargs):
        eos_token = kwargs.get('eos_token', None)
        num_beams = kwargs.get('num_beams', 3)
        length_penalty = kwargs.get('length_penalty', 0.7)
        temperature = kwargs.get('temperature', 1.)
        num_tokens = kwargs.get('num_tokens', None)        
        pad_token = kwargs.get('pad_token', 0)
        batch_size, t = start_tokens.shape
        was_training = self.net.training
        self.net.eval()
        beam_scores = torch.zeros(
            (batch_size, num_beams)).to(start_tokens.device)
        beam_scores[:, 1:] = -1e9  # prevent the first time beam repeating
        beam_scores = beam_scores.view(-1)
        done = [False for _ in range(batch_size)]
        generated_hyps = [BeamHypotheses(
            num_beams,  length_penalty=length_penalty) for _ in range(batch_size)]
        input_ids = start_tokens.repeat(num_beams, 1)
        hidden = context[:, None].repeat(1, num_beams, 1, 1).view(
            batch_size*num_beams, context.shape[1], context.shape[2])
        cur_len = t
        while cur_len < seq_len:
            outputs = self.net(x=input_ids, context=hidden)
            next_token_logits = outputs[:, -1, :]
            scores = F.log_softmax(next_token_logits/temperature, dim=-1)
            # cumulative log(prob)
            next_scores = scores + beam_scores[:, None].expand_as(scores)
            next_scores = next_scores.view(batch_size, num_beams * num_tokens)
            next_scores, next_tokens = torch.topk(
                next_scores, 2*num_beams, dim=1, largest=True, sorted=True)
            next_batch_beam = []
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    next_batch_beam.extend([(0, pad_token, 0)] * num_beams)# (0,0,0)->(score, token id, beam id)
                    continue
                next_sent_beam = []
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[batch_idx], next_scores[batch_idx])):
                    beam_id = beam_token_id // num_tokens  
                    token_id = beam_token_id % num_tokens 
                    effective_beam_id = batch_idx * num_beams + beam_id
                    if (eos_token is not None) and (token_id.item() == eos_token):
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),)
                    else:
                        next_sent_beam.append(
                            (beam_token_score, token_id, effective_beam_id))
                    if len(next_sent_beam) == num_beams:
                        break
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len)
                next_batch_beam.extend(next_sent_beam)
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            input_ids = input_ids[beam_idx, ...]
            hidden = hidden[beam_idx, ...]
            input_ids = torch.cat(
                [input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        sent_lengths = input_ids.new(output_batch_size)
        best = []
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, seq_len)
            decoded = input_ids.new(
                output_batch_size, sent_max_len).fill_(pad_token)
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < seq_len:
                    decoded[i, sent_lengths[i]] = eos_token
        else:
            decoded = torch.stack(best).type(torch.long)
        self.net.train(was_training)
        decoded = decoded[:, t:]
        return decoded

    @torch.no_grad()
    def generate(self, start_tokens, context, seq_len=256, filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        eos_token = kwargs.get('eos_token', None)
        temperature = kwargs.get('temperature', 1.)
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            # print('arw:', out.shape)
            logits = self.net(x, mask=mask, context=context)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out


def get_decoder(args):
    return CustomARWrapper(
        TransformerWrapper(
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            attn_layers=Decoder(
                dim=args.dim,
                depth=args.num_layers,
                heads=args.heads,
                **args.decoder_args
            )),
        pad_value=args.pad_token)
