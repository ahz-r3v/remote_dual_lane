"""Bounded CPU lifecycle evidence; no GPU timer, stream change or tensor ownership.

Installed as highway_dynamic_lifecycle in a NEW frozen engine. Wall timestamps
are host-visible launch/result/transfer boundaries, not GPU kernel timestamps.
Transfer byte counters wrap the existing synchronous API, never add a sync.
"""
from collections import deque
import copy
import os
import threading
import time
import uuid


class Journal:
    def __init__(self, limit=2048):
        self.lock = threading.RLock()
        self.events = deque(maxlen=limit)
        self.sequence = 0
        self.instance = uuid.uuid4().hex
        self.rooms = {}
        self.p_batches = {}
        self.transfers = {}
        self.local = threading.local()

    def emit(self, kind, **fields):
        with self.lock:
            self.sequence += 1
            event = dict(kind=kind, epoch=time.time(), sequence=self.sequence, **fields)
            self.events.append(event)
            return event

    def export(self):
        with self.lock:
            return copy.deepcopy(dict(schema=3, instance=self.instance,
                sequence=self.sequence, oldest_sequence=self.events[0]['sequence'] if self.events else 1,
                dropped_total=max(0, self.sequence-len(self.events)), events=list(self.events),
                inflight_prefill=list(self.p_batches.values()),
                inflight_transfers=list(self.transfers.values())))


journal = Journal()


def enabled():
    return os.environ.get('HIGHWAY_DYNAMIC_LIFECYCLE') == '1'


def tracked(req):
    return enabled() and str(req.rid).startswith('dynamic-')


def request_event(req, kind, **fields):
    if not tracked(req):
        return
    journal.emit(kind, rid=str(req.rid), **fields)


def queue_features(scheduler):
    return dict(batch=len(getattr(getattr(scheduler, 'running_batch', None), 'reqs', [])),
                queued=len(getattr(scheduler, 'waiting_queue', [])))


def register(req, mode, retracted=False):
    if not tracked(req):
        return
    room = getattr(req, 'bootstrap_room', None)
    with journal.lock:
        # Bounded scalar mapping, no Req/KV pool references. Trial IDs are unique.
        if room is not None:
            journal.rooms[str(room)] = str(req.rid)
            if len(journal.rooms) > 4096:
                del journal.rooms[next(iter(journal.rooms))]
    request_event(req, 'retracted' if retracted else 'registered', mode=str(mode))


def launch(scheduler, batch):
    if not enabled():
        return
    mode = getattr(scheduler.disaggregation_mode, 'value', str(scheduler.disaggregation_mode))
    if batch.forward_mode.is_decode():
        for req in batch.reqs:
            if tracked(req) and not getattr(req, '_dynamic_first_decode_launched', False):
                req._dynamic_first_decode_launched = True
                request_event(req, 'd_first_launch', batch_id=batch.forward_iter,
                              generated=len(req.output_ids))
        return
    if str(mode).lower() != 'prefill' or not batch.forward_mode.is_extend():
        return
    lengths = batch.seq_lens_cpu
    if getattr(getattr(lengths, 'device', None), 'type', None) != 'cpu':
        raise RuntimeError('Lifecycle observation requires existing CPU lengths')
    members = [dict(rid=str(req.rid), prefix=int(total)-int(n), tokens=int(n))
        for req, total, n in zip(batch.reqs, lengths.tolist(), batch.extend_lens) if tracked(req)]
    if members:
        with journal.lock:
            event = journal.emit('p_launch', batch_id=batch.forward_iter, members=members)
            journal.p_batches[batch.forward_iter] = event


def prefill_result(batch):
    # AFTER the native next_token_ids.tolist(): no new device operation here.
    from sglang.srt.observability.highway_dynamic_state import completed_prefill
    completed_prefill(batch)
    if enabled():
        with journal.lock:
            launched = journal.p_batches.pop(batch.forward_iter, None)
            if launched:
                journal.emit('p_result', batch_id=batch.forward_iter, members=launched['members'])


def chunk_key(room, span):
    return f'{room}:{span.start}:{span.stop}'


def transfer_queued(room, span, last):
    if not enabled():
        return
    with journal.lock:
        rid = journal.rooms.get(str(room))
        if rid:
            journal.emit('kv_queued', rid=rid, key=chunk_key(room, span),
                page_start=span.start, page_end=span.stop, last=bool(last), concurrent=len(journal.transfers))


def transfer_begin(chunk):
    journal.local.transfer = None
    if not enabled():
        return
    with journal.lock:
        rid = journal.rooms.get(str(chunk.room))
        if rid:
            fields = dict(rid=rid, key=chunk_key(chunk.room, chunk.index_slice),
                page_start=chunk.index_slice.start, page_end=chunk.index_slice.stop,
                last=bool(chunk.is_last_chunk), bytes_started=0, bytes_completed=0,
                fragments=0, calls=0, errors=0, inflight_at_start=len(journal.transfers))
            event = journal.emit('kv_begin', **fields)
            context = dict(fields, epoch=event['epoch'])
            journal.transfers[context['key']] = context
            journal.local.transfer = context


def bind_transfer_context(fn):
    # Explicitly propagate scalar context into native per-layer pool threads.
    context = getattr(journal.local, 'transfer', None)
    if context is None:
        return fn
    def wrapped(*args, **kwargs):
        previous = getattr(journal.local, 'transfer', None)
        journal.local.transfer = context
        try:
            return fn(*args, **kwargs)
        finally:
            journal.local.transfer = previous
    return wrapped


def transfer_call(engine, session, sources, destinations, lengths):
    context = getattr(journal.local, 'transfer', None)
    if context is None:
        return engine.batch_transfer_sync(session, sources, destinations, lengths)
    size = sum(int(n) for n in lengths)
    with journal.lock:
        context['bytes_started'] += size
        context['fragments'] += len(lengths)
        context['calls'] += 1
    succeeded = False
    try:
        result = engine.batch_transfer_sync(session, sources, destinations, lengths)
        succeeded = result == 0
        return result
    finally:
        with journal.lock:
            if succeeded:
                context['bytes_completed'] += size
                context['payload_completed_epoch'] = time.time()
            else:
                context['errors'] += 1


def transfer_end(ok=True):
    context = getattr(journal.local, 'transfer', None)
    if context is None:
        return
    with journal.lock:
        fields = dict(context)
        start = fields.pop('epoch')
        journal.emit('kv_end', **fields, start_epoch=start,
            duration_ms=1000*(time.time()-start), success=bool(ok and not context['errors']))
        journal.transfers.pop(context['key'], None)
        journal.local.transfer = None


def snapshot(scheduler):
    from sglang.srt.observability.highway_dynamic_state import snapshot as legacy_snapshot
    state = legacy_snapshot(scheduler)
    # This order is the real waiting queue, NOT the union of queue memberships.
    state['ready_order'] = [str(r.rid) for r in getattr(scheduler, 'waiting_queue', []) if tracked(r)]
    state['lifecycle'] = journal.export()
    queue = getattr(scheduler, 'disagg_prefill_bootstrap_queue', None)
    if queue is not None:
        args = queue.kv_manager.kv_args
        state_types = [str(t).split('.')[-1].lower() for t in getattr(args, 'state_types', [])]
        if any(t != 'mamba' for t in state_types):
            state['lifecycle']['unsupported_geometry'] = state_types
        else:
            extra = [int(n) for group in getattr(args, 'state_item_lens', []) for n in group]
            aux = [int(n) for n in args.aux_item_lens]
            state['lifecycle']['geometry'] = dict(page_tokens=int(args.page_size),
                bytes_per_page=sum(int(n) for n in args.kv_item_lens),
                last_state_aux_bytes=sum(extra+aux), last_fragments=len(extra+aux))
    # Publish after the CPU journal copy; every event source time <= state epoch.
    state['epoch'] = time.time()
    return state
