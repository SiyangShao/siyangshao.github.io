+++ 
draft = false 
date= 2023-07-20T02:39:53+08:00
title = "PCIe Stress Test"
description = ""
slug = ""
authors = []
tags = ['system']
categories = ['research']
externalLink = ""
series = []
+++
# Monitor PCIe Stress
Technically, there's no much way to monitor the current PCIe bandwidth occupation. In most situations, the easiest way is to use 
``` bash
sudo lspci -vvv
```
to check the current PCIe link speed. However, it's not real-time (it will only tell you the PCIe link cap, like PCIe 4.0 x16 ).

Luckily, if you are using the NVIDIA GPU, you can use 
``` bash
nvidia-smi dmon -s t
```
to monitor the current PCIe bandwidth occupation. It will call a low level api
``` bash
nvmlDeviceGetPcieThroughput
```
to get the current PCIe bandwidth occupation. ```nvidia-smi``` only support minimum interval of 1s, but this api is querying a byte counter over a 20ms interval. For more detailed information, you can check NVIDIA's document [here](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1gd86f1c74f81b5ddfaa6cb81b51030c72).


# Bandwidth Test - Stressor
For indirectly measure the bandwidth, NVIDIA also provides something called "Bandwidthtest" which you can find on [github](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/bandwidthTest). Technically, this program is just contionusly copy data between CPU and GPU, using ```cudaMemcpyAsync```, and calculate the bandwidth. You can manually change the transfer size and number of iteration.

Thus, this program can be changed to apply stress on PCIe. For example, we can use the default transfer size, but change the number of iterations to 1e7, which will occupy the PCIe for a long time. Then, we can run both host to device transfer and device to host transfer at the same time (as PCIe is full duplex).

# Stress vs Model Latency
In general, we run multiple stressor process in parallel with the model to check whether the latency has changed. The result was obvious that running more stressor will cause higher latency and more significant tail latency.

But, it's somewhat strange that if only running the stressor, the total bandwidth usage does not have a large difference. So we guess, the result may be caused by PCIe allocation policy.

For example, if the policy is FIFO or Round-Robin, more stressor process will result in more PCIe transfer packets, which will lead to a higher latency. However, if the policy is priority based, the stressor process may not have a large impact on the latency.
