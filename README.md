![GitHub Actions](https://img.shields.io/github/actions/workflow/status/p-doom/nvidia_gpu_prometheus_exporter/release.yml?branch=master&label=CI&logo=github&style=flat-square)


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
