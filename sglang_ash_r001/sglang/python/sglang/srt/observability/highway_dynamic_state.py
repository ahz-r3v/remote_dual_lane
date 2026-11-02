"""CPU-only scheduler snapshot. No Torch import, GPU reads, or synchronization."""
import time


def completed_prefill(batch):
    lengths=batch.seq_lens_cpu
    if lengths is None or getattr(getattr(lengths,'device',None),'type',None)!='cpu':
        raise RuntimeError('Dynamic observation requires existing CPU sequence lengths')
    # Called AFTER the existing native result-copy completion, not after scheduling.
    for req,length in zip(batch.reqs,lengths.tolist()):
        req._dynamic_completed_prefill=max(getattr(req,'_dynamic_completed_prefill',0),
                                         min(int(length),len(req.origin_input_ids)))
        req._dynamic_prefill_result_epoch=time.time()


def snapshot(scheduler):
    start=time.perf_counter();epoch=time.time();items={}
    mode=getattr(scheduler.disaggregation_mode,'value',str(scheduler.disaggregation_mode))
    def add(reqs,stage,priority):
        for wrapped in reqs or []:
            req=getattr(wrapped,'req',wrapped)
            rid=str(req.rid)
            if not rid.startswith('dynamic-'):continue
            item=items.get(rid)
            if item is not None:
                item['memberships'].append(stage)
                if priority<=item['_priority']:continue
                memberships=item['memberships']
            else:memberships=[stage]
            generated=len(req.output_ids);budget=int(req.sampling_params.max_new_tokens)
            items[rid]=dict(rid=rid,stage=stage,memberships=memberships,_priority=priority,
                input_tokens=len(req.origin_input_ids),output_budget=budget,
                generated_tokens=generated,prefill_completed_tokens=getattr(req,'_dynamic_completed_prefill',0),
                prefill_result_epoch=getattr(req,'_dynamic_prefill_result_epoch',None),
                scheduled_fill_tokens=int(getattr(req,'fill_len',0)),
                output_remaining=max(0,budget-generated),
                req_finished=bool(req.finished()))
    for attr,stage in [('disagg_prefill_bootstrap_queue','p_bootstrap'),
                       ('disagg_decode_prealloc_queue','d_prealloc'),
                       ('disagg_decode_transfer_queue','d_transfer')]:
        queue=getattr(scheduler,attr,None)
        if queue is not None:add(getattr(queue,'queue',[]),stage,1)
    add(getattr(scheduler,'waiting_queue',[]),'waiting',2)
    if getattr(scheduler,'chunked_req',None) is not None:add([scheduler.chunked_req],'p_chunked',3)
    batches=[]
    for name in ['running_batch','last_batch','cur_batch']:
        batch=getattr(scheduler,name,None)
        if batch is None:continue
        reqs=getattr(batch,'reqs',[])
        add(reqs,'batch_visible',4)
        batches.append(dict(source=name,request_ids=[r.rid for r in reqs if str(r.rid).startswith('dynamic-')],
                            forward_mode=str(getattr(batch,'forward_mode','unknown'))))
    for pair in getattr(scheduler,'result_queue',[]) or []:
        add(getattr(pair[0],'reqs',[]),'pending_result',5)
    add(getattr(scheduler,'disagg_prefill_inflight_queue',[]),'p_transfer',6)
    for item in items.values():item.pop('_priority')
    scheduler._dynamic_snapshot_seq=getattr(scheduler,'_dynamic_snapshot_seq',0)+1
    return dict(schema=1,sequence=scheduler._dynamic_snapshot_seq,epoch=epoch,mode=str(mode),
                requests=list(items.values()),batches=batches,
                build_ms=1000*(time.perf_counter()-start),
                semantics='CPU-result-visible progress; overlapping GPU may be ahead; not a globally atomic snapshot')
