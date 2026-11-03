"""Dump real full-attention KV (Qwen3.6, real weights, real text) for offline lossless-compressibility analysis.
Real text = ShareGPT prompts (datasets/sharegpt/extract1000.json, prompt_ids decoded with the extract's tokenizer), re-tokenized with the model's own tokenizer and
concatenated into N_SEQ sequences of SEQ_LEN tokens; prefilled with SGLang's ModelRunner (bench_one_batch style, as the harness).
Saves per sequence: K and V of the 10 full-attention layers, bf16, [layers, tokens, kv_heads, head_dim] -> <out>/kv_seq<i>.pt.
Usage (inside gpuguard, env.sh sourced): python kv_dump.py --model-path ... --out runs/kv_dump [--seq-len 16384 --n-seq 2]"""
import argparse, json, os, sys
from array import array
import torch

ap0 = argparse.ArgumentParser(add_help=False)
ap0.add_argument('--out', required=True); ap0.add_argument('--seq-len', type=int, default=16384); ap0.add_argument('--n-seq', type=int, default=2)
ap0.add_argument('--data', default='/mnt/vdb/chengzhi/datasets/sharegpt/extract1000.json')
own, rest = ap0.parse_known_args()

import sglang.bench_one_batch as bob
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def token_streams(tok, n, L):
    """n sequences of exactly L tokens from consecutive ShareGPT prompts (each prompt followed by a blank line)."""
    d = json.load(open(own.data)); entries = d['entries']
    # 'prompt' is a truncated preview; the full text only exists as prompt_ids of the tokenizer the extract was made with (Qwen3-8B),
    # so decode with that tokenizer and re-encode with this model's tokenizer
    from transformers import AutoTokenizer
    src = AutoTokenizer.from_pretrained(d['tokenizer'])
    seqs, cur = [], []
    for e in entries:
        cur += tok.encode(src.decode(e['prompt_ids']) + '\n\n', add_special_tokens=False)
        while len(cur) >= L and len(seqs) < n:
            seqs.append(cur[:L]); cur = cur[L:]
        if len(seqs) == n:
            break
    assert len(seqs) == n, f'not enough text for {n} x {L} tokens'
    return seqs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p); sa = ServerArgs.from_cli_args(p.parse_args(rest))
    os.makedirs(own.out, exist_ok=True)
    runner, tok = bob.load_model(sa, PortArgs.init_new(sa), 0, 0)
    runner = runner.torch_runner if hasattr(runner, 'torch_runner') else runner
    seqs = token_streams(tok, own.n_seq, own.seq_len)
    pool = runner.token_to_kv_pool
    full = pool.full_kv_pool if hasattr(pool, 'full_kv_pool') else pool
    full_ids = list(getattr(pool, 'full_attention_layer_ids', range(len(full.k_buffer))))
    print(json.dumps(dict(full_attention_layer_ids=full_ids, k_buffer_shape=list(full.k_buffer[0].shape))), flush=True)
    sp = SamplingParams(temperature=0, max_new_tokens=1)
    for i, ids in enumerate(seqs):
        r = Req(rid=f'dump{i}', origin_input_text='', origin_input_ids=array('q', ids), sampling_params=sp)
        r.full_untruncated_fill_ids = r.origin_input_ids; r.fill_len = len(ids); r.logprob_start_len = -1
        r.set_extend_input_len(r.fill_len - len(r.prefix_indices))
        tc = bob.TreeCacheNamespace(page_size=runner.server_args.page_size, device=runner.device,
                                    token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator)
        b = ScheduleBatch.init_new(reqs=[r], req_to_token_pool=runner.req_to_token_pool, token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
                                   tree_cache=tc, model_config=runner.model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
        b.prepare_for_extend()
        if b.input_ids is None and getattr(b, 'prefill_input_ids_cpu', None) is not None:
            b.input_ids = b.prefill_input_ids_cpu.to(b.device, non_blocking=True); b.prefill_input_ids_cpu = None
        runner.forward(ForwardBatch.init_new(b, runner)); torch.cuda.synchronize()
        loc = runner.req_to_token_pool.req_to_token[b.req_pool_indices[0], :len(ids)].long()
        K = torch.stack([full.k_buffer[j][loc] for j in range(len(full.k_buffer))]).cpu()   # [layers, tokens, kv_heads, head_dim]
        V = torch.stack([full.v_buffer[j][loc] for j in range(len(full.v_buffer))]).cpu()
        torch.save(dict(K=K, V=V, layer_ids=full_ids, tokens=ids, source='sharegpt extract1000 prompts'), os.path.join(own.out, f'kv_seq{i}.pt'))
        print(json.dumps(dict(seq=i, tokens=len(ids), K=list(K.shape), dtype=str(K.dtype), K_absmax=float(K.float().abs().max()),
                              V_absmax=float(V.float().abs().max()))), flush=True)


if __name__ == '__main__':
    main()
