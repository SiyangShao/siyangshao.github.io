+++
draft = false
date = 2024-02-18T21:36:04+08:00
title = "Continuous Batching 代码简介"
description = ""
slug = ""
authors = []
tags = ['system', 'mlsys']
categories = ['research']
externalLink = ""
series = []
+++

> 这篇文章也相当于我这几天读vLLM源码的一个总结. 如果发现有事实错误等方面, 请及时联系我.

# Continuous Batching 介绍

Continuous Batching的核心思想是, 在传统的批处理(即 static batching)的过程中, 由于我们无法预测每一个 sequence需要多久才能结束, 导致如果不同的sequence结束的token差越大时, 会导致GPU的利用率偏低. 在一个serving的后期, 除了还没有结束的sequence在计算next token, 剩下的已经结束的sequence相当于empty token在空转.

所以, continuous batching选择了迭代处理方式, 在部分序列处理完成后, 选择插入新序列. 这样能提高利用率. 关于这个idea不清楚的可以去参考[这篇文章](https://www.usenix.org/conference/osdi22/presentation/yu).

# vLLM生成

对vLLM而言, 他添加了一个调度器(即代码中的 scheduler), 而所有生成都会由调度器处理. 所以在generate中, 我们可以发现, 他是将所有的request(即在同一个batch中的prompt)都放入调度器进行处理的.

``` python
num_requests = len(prompts) if prompts is not None else len(prompt_token_ids)
for i in range(num_requests):
	prompt = prompts[i] if prompts is not None else None
    token_ids = None if prompt_token_ids is None else prompt_token_ids[i]
    self._add_request(prompt,token_ids)
return self._run_engine()
```

```python
def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
	# Run the engine.
	outputs: List[RequestOutput] = []
	while self.llm_engine.has_unfinished_requests():
		step_outputs = self.llm_engine.step()
		for output in step_outputs:
			if output.finished:
				outputs.append(output)
   	outputs = sorted(outputs, key=lambda x: int(x.request_id))
	return outputs
```

而调度器会在每次调用step的时候进行处理

``` python
seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
if not scheduler_outputs.is_empty():
# Execute the model.
	all_outputs = self._run_workers(
		"execute_model",
		driver_kwargs={
			"seq_group_metadata_list": seq_group_metadata_list,
			"blocks_to_swap_in": scheduler_outputs.blocks_to_swap_in,
			"blocks_to_swap_out": scheduler_outputs.blocks_to_swap_out,
			"blocks_to_copy": scheduler_outputs.blocks_to_copy,
			})
# Only the driver worker returns the sampling results.
	output = all_outputs[0]
else:
	output = []
return self._process_model_outputs(output, scheduler_outputs)
```

在这里我们先忽略如何处理output, 重点关注两个部分: 调度器如何运作, 以及如何处理.

首先让我们先忽略调度部分, 看一下是如何推理的. 在```self._run_workers```部分中, 较为核心的部分就是他如何在本地和ray调用```execute_model``` method的. 我们目前只关心本地部分.

可以发现, ```worker```的```execute model```其实就是如下: 

``` python
self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)
output = self.model_runner.execute_model(seq_group_metadata_list, self.gpu_cache)
```

首先, 将scheduler得到的需要swap的block给读入, 然后进行处理.

而真正的```execute_model```部分位于```model_runner``中, 如下:

``` python
def execute_model(
    self,
    seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Optional[SamplerOutput]:
    input_tokens, input_positions, input_metadata, sampling_metadata, lora_requests, lora_mapping = (
        self.prepare_input_tensors(seq_group_metadata_list))

    if self.lora_config:
        self.set_active_loras(lora_requests, lora_mapping)

    # Execute the model.
    if input_metadata.use_cuda_graph:
        graph_batch_size = input_tokens.shape[0]
        model_executable = self.graph_runners[graph_batch_size]
    else:
        model_executable = self.model
    hidden_states = model_executable(
        input_ids=input_tokens,
        positions=input_positions,
        kv_caches=kv_caches,
        input_metadata=input_metadata,
    )

    # Sample the next token.
    output = self.model.sample(
        hidden_states=hidden_states,
        sampling_metadata=sampling_metadata,
    )
    return output
```

这里设置了模型的hidden state和采样策略, 最后生成下一个token.

然后是scheduler的调度部分

```python
def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
# Schedule sequence groups.
# This function call changes the internal states of the scheduler
# such as self.running, self.swapped, and self.waiting.
	scheduler_outputs = self._schedule()
# Create input data structures.
	seq_group_metadata_list: List[SequenceGroupMetadata] = []
	for seq_group in scheduler_outputs.scheduled_seq_groups:
		seq_data: Dict[int, SequenceData] = {}
		block_tables: Dict[int, List[int]] = {}
		for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
			seq_id = seq.seq_id
			seq_data[seq_id] = seq.data
			block_tables[seq_id] = self.block_manager.get_block_table(seq)
		seq_group_metadata = SequenceGroupMetadata(
			request_id=seq_group.request_id,
			is_prompt=scheduler_outputs.prompt_run,
			seq_data=seq_data,
			sampling_params=seq_group.sampling_params,
			block_tables=block_tables,
			lora_request=seq_group.lora_request,
			prefix=seq_group.prefix,
			)
		seq_group_metadata_list.append(seq_group_metadata)
return seq_group_metadata_list, scheduler_outputs
```

在```self._schedule()```中, 则是更新当前状态. 简单来说, 他会首先检查当前的slot是否足够, 且没有swap out的sequence. 如果有swap out的sequence, 第一选择是extend slot直到无法扩充. 在无法扩充时, 其实现了一个抢占性的调度策略.

在generate过程中, 每一个sequence都只会占用一个token slot. 因此, batched token的数量永远等于处于running state的sequence数量.
