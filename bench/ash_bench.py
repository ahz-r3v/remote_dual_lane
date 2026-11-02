"""sglang.bench_one_batch with chunked prefill (h100_decode_bw_r001).
Only change: extend() prefills the requests in groups of <= ASH_PREFILL_TOKENS tokens (default 32768; a longer single
request is its own group) and merges the groups with ScheduleBatch.merge_batch (the scheduler's own running-batch merge),
so large B x ctx points do not OOM / hit kernel launch limits in one giant prefill. Decode timing and the profiler range
(--profile-activities CUDA_PROFILER, cudaProfilerStart/Stop around the chosen decode steps) are bench_one_batch's own.
"""
import argparse, logging, os
import torch
import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs

TOK = int(os.environ.get('ASH_PREFILL_TOKENS', '32768'))
_orig_extend = bob.extend


def extend_chunked(reqs, model_runner):
    groups, cur, n = [], [], 0
    for r in reqs:
        L = len(r.origin_input_ids)
        if cur and n + L > TOK:
            groups.append(cur)
            cur, n = [], 0
        cur.append(r)
        n += L
    groups.append(cur)
    ids, logits, batch = _orig_extend(groups[0], model_runner)
    for g in groups[1:]:
        i2, l2, b2 = _orig_extend(g, model_runner)
        batch.merge_batch(b2)
        ids, logits = torch.cat([ids, i2]), torch.cat([logits, l2])
    return ids, logits, batch


bob.extend = extend_chunked

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    bob.BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = bob.BenchArgs.from_cli_args(args)
    logging.basicConfig(level=getattr(logging, server_args.log_level.upper()), format='%(message)s')
    bob.main(server_args, bench_args)
