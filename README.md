![GitHub Actions](https://img.shields.io/github/actions/workflow/status/p-doom/nvml_exporter/release.yml?branch=master&label=CI&logo=github)


NVIDIA GPU Prometheus Exporter
------------------------------

This is a [Prometheus Exporter](https://prometheus.io/docs/instrumenting/exporters/) for
exporting NVIDIA GPU metrics. It uses the [NVIDIA Go NVML bindings](github.com/NVIDIA/go-nvml)
for [NVIDIA Management Library](https://developer.nvidia.com/nvidia-management-library-nvml)
(NVML) which is a C-based API that can be used for monitoring NVIDIA GPU devices.
Unlike some other similar exporters, it does not call the
[`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) binary.

This Exporter is a fork of https://github.com/mindprince/nvidia_gpu_prometheus_exporter with the following main changes:
- added parsing of /run/gpustat/XX for jobid and uid of the user running on the GPU. Slurm scripts that take advantage of this are available on [jobstats website](https://github.com/PrincetonUniversity/jobstats).
- switched from [Go bindings](https://github.com/mindprince/gonvml) to [NVIDIA Go NVML bindings](github.com/NVIDIA/go-nvml)
- added support for MIG instance autodetection and stats
- for Hopper and newer GPUs it now also provides [GPM](https://docs.nvidia.com/deploy/nvml-api/group__GPM.html) based metrics, like SM utilization, occupancy and a few others. To disable these metrics use `-disable.gpm` option.
## Building

E.g.
```
go build
```

## Running

The exporter requires the following:
- access to NVML library (`libnvidia-ml.so.1`).
- access to the GPU devices.

To make sure that the exporter can access the NVML libraries, either add them
to the search path for shared libraries. Or set `LD_LIBRARY_PATH` to point to
their location.

By default the metrics are exposed on `localhost:9445/metrics`. The port can be
modified using the `-web.listen-address` flag.

## Alerts
The collector allows us to alert on some of the bad states the GPU might be in. The following are ones we currently use:
```
  - alert: GPUError
    expr: nvidia_gpu_last_error > 0
    for: '0m'
    labels:
      severity: warning
      alerttype: hardware
    annotations:
      summary: 'GPU{{ $labels.minor_number }} error while collecting data on {{ reReplaceAll ":.*" "" $labels.instance }}'
      description: 'Host {{ reReplaceAll ":.*" "" $labels.instance }} had error {{ printf "%.f" $value }} while fetching GPU metrics from GPU{{ $labels.minor_number }}'
  - alert: GPUECCError
    expr: nvidia_gpu_ecc_errors{counter="volatile",error="uncorrected"} > 0
    for: '0m'
    labels:
      severity: warning
      alerttype: hardware
    annotations:
      summary: 'GPU{{ $labels.minor_number }} Uncorrected ECC error on {{ reReplaceAll ":.*" "" $labels.instance }}'
      description: 'Host {{ reReplaceAll ":.*" "" $labels.instance }} has {{ printf "%.f" $value }} uncorrected ECC errors on GPU{{ $labels.minor_number }}'
  - alert: GPUClockError
    expr: nvidia_gpu_clock_event_reason{reason=~".*slowdown.*"} > 31
    for: '5m'
    labels:
      severity: warning
      alerttype: hardware
    annotations:
      summary: 'GPU{{ $labels.minor_number }} Clock Problem {{ $labels.reason }} on {{ reReplaceAll ":.*" "" $labels.instance }}'
      description: 'Host {{ reReplaceAll ":.*" "" $labels.instance }} has {{ $labels.reason }} clock error on GPU{{ $labels.minor_number }}'
```
