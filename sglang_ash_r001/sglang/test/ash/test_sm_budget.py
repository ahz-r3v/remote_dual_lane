"""Unit tests for the ASH r001 green-context lane changes (CPU only; run: python test/ash/test_sm_budget.py, or pytest).
Covers: FA3 lane split choice, the virtual SM count vs FA3's dynamic split rule, the long_green_ctx_sm_count = -1 sentinel, the
SM-count-specific fused-MoE config file names, and the sm_budget context (without a GPU, by faking the device SM count)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))

import sglang.srt.multiplex.sm_budget as smb  # noqa: E402


def fa3_dynamic_splits(n_blocks, batch, kv_heads, num_sm, num_splits_cap):
    """FA3 scheduler-metadata prepare rule (flash_prepare_scheduler.cu) for equal-length decode requests."""
    total_blocks = n_blocks * batch
    blocks_per_sm = math.ceil(total_blocks * 1.1 * kv_heads / num_sm)
    return max(min(math.ceil(n_blocks / blocks_per_sm), num_splits_cap), 1)


def test_choose_lane_num_splits():
    from sglang.srt.layers.attention.flashattention_backend import choose_lane_num_splits
    assert choose_lane_num_splits(64, 128) == 2
    assert choose_lane_num_splits(64, 112) == 7      # 448 tiles = 4 full waves
    assert choose_lane_num_splits(64, 96) == 3       # 192 tiles = 2 full waves
    assert choose_lane_num_splits(32, 32) == 1
    assert choose_lane_num_splits(32, 16) == 1
    for tiles in range(1, 200):
        for n in (16, 32, 64, 96, 112, 128):
            s = choose_lane_num_splits(tiles, n)
            assert 1 <= s <= 16


def test_virtual_num_sm_hits_target_split():
    from sglang.srt.layers.attention.flashattention_backend import choose_lane_num_splits, fa3_metadata_num_sm
    for batch in (1, 4, 8, 16, 32, 48, 64):
        for kv in (1, 2, 4, 8):
            for n in (96, 112, 128):
                tiles = batch * kv
                s = choose_lane_num_splits(tiles, n)
                virt = fa3_metadata_num_sm(tiles, s, n)
                for ctx in (4096, 16384, 32768, 51200, 131072):
                    n_blocks = math.ceil(ctx / 128)
                    if n_blocks < s:
                        continue
                    got = fa3_dynamic_splits(n_blocks, batch, kv, virt, s)
                    if n_blocks >= 2 * s:     # long enough for the split to matter: must hit the target exactly
                        assert got == s, (batch, kv, n, ctx, s, virt, got)
                    else:                     # short contexts: the dynamic rule may use fewer splits (never more)
                        assert 1 <= got <= s, (batch, kv, n, ctx, s, virt, got)


def test_long_lane_remainder_sentinel():
    from sglang.srt.multiplex.decode_lane_context import _resolve_green_ctx_sm_counts
    assert _resolve_green_ctx_sm_counts(128, -1, 132, 8) == (128, 4)
    assert _resolve_green_ctx_sm_counts(96, -1, 132, 8) == (96, 36)
    assert _resolve_green_ctx_sm_counts(96, 32, 132, 8) == (96, 32)    # unchanged behaviour
    for bad in ((0, -1), (132, -1), (100, -1)):
        try:
            _resolve_green_ctx_sm_counts(bad[0], bad[1], 132, 8)
            raise AssertionError(f'expected ValueError for {bad}')
        except ValueError:
            pass


def test_moe_config_file_names():
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config as c
    c.get_device_name = lambda: 'NVIDIA H200'
    full = c.get_config_file_name(256, 512, 'fp8_w8a8', [128, 128])
    lane = c.get_config_file_name(256, 512, 'fp8_w8a8', [128, 128], sm_count=96)
    down = c.get_config_file_name(256, 512, 'fp8_w8a8', [128, 128], down_moe=True, sm_count=96)
    assert full == 'E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json', full
    assert lane == 'E=256,N=512,device_name=NVIDIA_H200,sms=96,dtype=fp8_w8a8,block_shape=[128, 128].json', lane
    assert down.endswith('block_shape=[128, 128]_down.json') and ',sms=96,' in down


def test_sm_budget_context():
    smb.full_sm_count.cache_clear()
    smb.full_sm_count = lambda device=None: 132          # no GPU needed
    assert smb.current_sm_budget() == 132 and not smb.is_partial_budget()
    with smb.sm_budget(96):
        assert smb.current_sm_budget() == 96 and smb.is_partial_budget()
        with smb.sm_budget(None):                         # no-op inside
            assert smb.current_sm_budget() == 96
    assert smb.current_sm_budget() == 132
    with smb.sm_budget(132):                              # full device = no budget
        assert not smb.is_partial_budget()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); print('ok', name)
