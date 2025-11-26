package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	namespace = "nvidia_gpu"

	GPM_DISABLED        = 0
	GPM_NO_DATA         = 1
	GPM_SAMPLE1_ONLY    = 2
	GPM_SAMPLE1_SAMPLE2 = 3
	GPM_SAMPLE2_SAMPLE1 = 4
)

var (
	addr                   = flag.String("web.listen-address", ":9445", "Address to listen on for web interface and telemetry.")
	disableExporterMetrics = flag.Bool("web.disable-exporter-metrics", false, "Exclude metrics about the exporter itself (promhttp_*, process_*, go_*)")
	disableGpm             = flag.Bool("disable.gpm", false, "Disable GPM metrics (which are available on Hopper and newer GPUs).")

	labels         = []string{"ordinal", "minor_number", "uuid", "name", "GPU_I_ID"}
	labelsJobInfo  = []string{"ordinal", "minor_number", "uuid", "name", "GPU_I_ID", "jobid", "userid"}
	eccLabels      = []string{"ordinal", "minor_number", "uuid", "name", "error", "counter"}
	eccErrorType   = []string{nvml.MEMORY_ERROR_TYPE_CORRECTED: "corrected", nvml.MEMORY_ERROR_TYPE_UNCORRECTED: "uncorrected"}
	eccCounterType = []string{nvml.VOLATILE_ECC: "volatile", nvml.AGGREGATE_ECC: "aggregate"}

	gpmState    = make(map[string]int)
	gpmSamples1 = make(map[string]nvml.GpmSample)
	gpmSamples2 = make(map[string]nvml.GpmSample)
)

type Collector struct {
	sync.Mutex
	numDevices          prometheus.Gauge
	usedMemory          *prometheus.GaugeVec
	totalMemory         *prometheus.GaugeVec
	dutyCycle           *prometheus.GaugeVec
	powerUsage          *prometheus.GaugeVec
	temperature         *prometheus.GaugeVec
	fanSpeed            *prometheus.GaugeVec
	eccErrors           *prometheus.GaugeVec
	lastError           *prometheus.GaugeVec
	jobId               *prometheus.GaugeVec
	jobUid              *prometheus.GaugeVec
	graphicsUtil        *prometheus.GaugeVec
	smUtil              *prometheus.GaugeVec
	smOccupancy         *prometheus.GaugeVec
	integerUtil         *prometheus.GaugeVec
	anyTensorUtil       *prometheus.GaugeVec
	dramBwUtil          *prometheus.GaugeVec
	fp64Util            *prometheus.GaugeVec
	fp32Util            *prometheus.GaugeVec
	fp16Util            *prometheus.GaugeVec
	pcieTxPerSec        *prometheus.GaugeVec
	pcieRxPerSec        *prometheus.GaugeVec
	nvlinkTotalRxPerSec *prometheus.GaugeVec
	nvlinkTotalTxPerSec *prometheus.GaugeVec
}

type Device struct {
	name       string
	uuid       string
	ordinal    string
	instanceId string
	device     nvml.Device
}

func NewCollector() *Collector {
	return &Collector{
		numDevices: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "num_devices",
				Help:      "Number of GPU devices",
			},
		),
		usedMemory: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "memory_used_bytes",
				Help:      "Memory used by the GPU device in bytes",
			},
			labelsJobInfo,
		),
		totalMemory: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "memory_total_bytes",
				Help:      "Total memory of the GPU device in bytes",
			},
			labelsJobInfo,
		),
		dutyCycle: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "duty_cycle",
				Help:      "Percent of time over the past sample period during which one or more kernels were executing on the GPU device",
			},
			labelsJobInfo,
		),
		powerUsage: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "power_usage_milliwatts",
				Help:      "Power usage of the GPU device in milliwatts",
			},
			labelsJobInfo,
		),
		temperature: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "temperature_celsius",
				Help:      "Temperature of the GPU device in celsius",
			},
			labelsJobInfo,
		),
		fanSpeed: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "fanspeed_percent",
				Help:      "Fanspeed of the GPU device as a percent of its maximum",
			},
			labelsJobInfo,
		),
		eccErrors: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "ecc_errors",
				Help:      "ECC Errors, with labels describing what they are",
			},
			eccLabels,
		),
		lastError: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "last_error",
				Help:      "Last error returned while trying to get stats from this GPU",
			},
			labels,
		),
		jobId: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "jobId",
				Help:      "JobId number of a job currently using this GPU as reported by Slurm",
			},
			labelsJobInfo,
		),
		jobUid: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "jobUid",
				Help:      "Uid number of user running jobs on this GPU",
			},
			labelsJobInfo,
		),
		graphicsUtil: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "graphics_util_percent",
				Help:      "Percentage of time any compute/graphics app was active on the GPU",
			},
			labelsJobInfo,
		),
		smUtil: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "sm_util_percent",
				Help:      "Percentage of SMs that were busy",
			},
			labelsJobInfo,
		),
		smOccupancy: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "sm_occupancy_percent",
				Help:      "Percentage of warps that were active vs theoretical maximum",
			},
			labelsJobInfo,
		),
		integerUtil: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "integer_util",
				Help:      "Percentage of time the GPU's SMs were doing integer operations",
			},
			labelsJobInfo,
		),
		anyTensorUtil: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "any_tensor_util_percent",
				Help:      "Percentage of time the GPU's SMs were doing ANY tensor operations",
			},
			labelsJobInfo,
		),
		dramBwUtil: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "dram_bw_util_percent",
				Help:      "Percentage of DRAM bw used vs theoretical maximum",
			},
			labelsJobInfo,
		),
		fp64Util: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "fp64_util_percent",
				Help:      "Percentage of time the GPU's SMs were doing non-tensor FP64 math",
			},
			labelsJobInfo,
		),
		fp32Util: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "fp32_util_percent",
				Help:      "Percentage of time the GPU's SMs were doing non-tensor FP32 math",
			},
			labelsJobInfo,
		),
		fp16Util: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "fp16_util_percent",
				Help:      "Percentage of time the GPU's SMs were doing non-tensor FP16 math",
			},
			labelsJobInfo,
		),
		pcieTxPerSec: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "pcie_tx_per_sec",
				Help:      "PCIe traffic from this GPU in bytes/sec",
			},
			labelsJobInfo,
		),
		pcieRxPerSec: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "pcie_rx_per_sec",
				Help:      "PCIe traffic to this GPU in bytes/sec",
			},
			labelsJobInfo,
		),
		nvlinkTotalRxPerSec: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "nvlink_total_rx_per_sec",
				Help:      "NvLink read bandwidth for all links in bytes/sec",
			},
			labelsJobInfo,
		),
		nvlinkTotalTxPerSec: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "nvlink_total_tx_per_sec",
				Help:      "NvLink write bandwidth for all links in bytes/sec",
			},
			labelsJobInfo,
		),
	}
}

func (c *Collector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.numDevices.Desc()
	c.usedMemory.Describe(ch)
	c.totalMemory.Describe(ch)
	c.dutyCycle.Describe(ch)
	c.powerUsage.Describe(ch)
	c.temperature.Describe(ch)
	c.fanSpeed.Describe(ch)
	c.lastError.Describe(ch)
}

func allocGpmSamples(uuid string) error {
	err := nvml.SUCCESS
	if _, exists := gpmSamples1[uuid]; !exists {
		gpmSamples1[uuid], err = nvml.GpmSampleAlloc()
		if err != nvml.SUCCESS {
			log.Printf("GPU(%s) failed to allocate GpmSample1 with error: %v", uuid, err)
			return err
		}
		gpmSamples2[uuid], err = nvml.GpmSampleAlloc()
		if err != nvml.SUCCESS {
			log.Printf("GPU(%s) failed to allocate GpmSample2 with error: %v", uuid, err)
			return err
		}
	}
	return err
}

func (c *Collector) Collect(ch chan<- prometheus.Metric) {
	// Only one Collect call in progress at a time.
	c.Lock()
	defer c.Unlock()

	c.usedMemory.Reset()
	c.totalMemory.Reset()
	c.dutyCycle.Reset()
	c.powerUsage.Reset()
	c.temperature.Reset()
	c.fanSpeed.Reset()
	c.lastError.Reset()

	numDevices, err := nvml.DeviceGetCount()
	if err != nvml.SUCCESS {
		log.Printf("DeviceCount() error: %v", err)
		return
	} else {
		c.numDevices.Set(float64(numDevices))
		ch <- c.numDevices
	}

	for i := 0; i < int(numDevices); i++ {
		//var lasterror = nvmlReturn_t = nvml.SUCCESS

		ordinal := strconv.Itoa(i)

		dev, err := nvml.DeviceGetHandleByIndex(i)
		if err != nvml.SUCCESS {
			log.Printf("DeviceHandleByIndex(%d) error: %v", i, err)
			continue
		}

		minorNumber, err := dev.GetMinorNumber()
		if err != nvml.SUCCESS {
			log.Printf("MinorNumber(%d) error: %v", i, err)
			c.lastError.WithLabelValues(ordinal, "", "", "", "").Set(float64(err))
			continue
		}
		minor := strconv.Itoa(int(minorNumber))

		uuid, err := dev.GetUUID()
		if err != nvml.SUCCESS {
			log.Printf("UUID(%s) error: %v", minor, err)
			c.lastError.WithLabelValues(ordinal, minor, "", "", "").Set(float64(err))
			continue
		}

		gpm, exists := gpmState[uuid]
		if !exists {
			if *disableGpm {
				gpm = GPM_DISABLED
			} else if gpuQuerySupport, ret := dev.GpmQueryDeviceSupport(); ret != nvml.SUCCESS {
				log.Printf("GPU(%s) failed to return gpm status, disabling, error: %v", uuid, ret)
				gpm = GPM_DISABLED
			} else {
				if gpuQuerySupport.IsSupportedDevice == 0 {
					gpm = GPM_DISABLED
				} else {
					gpm = GPM_NO_DATA
				}
				log.Printf("GPU(%s) gpm = %d", uuid, gpm)
			}
			if gpm != GPM_DISABLED {
				if allocGpmSamples(uuid) != nvml.SUCCESS {
					log.Printf("GPU(%s) disabling gpm due to GpmSample alloc errors", uuid)
					gpm = GPM_DISABLED
				}
			}
			gpmState[uuid] = gpm
		}

		name, err := dev.GetName()
		if err != nvml.SUCCESS {
			log.Printf("Name(%s) error: %v", minor, err)
			c.lastError.WithLabelValues(ordinal, minor, uuid, "", "").Set(float64(err))
			continue
		}

		currentMig, _, err := dev.GetMigMode()
		if err != nvml.SUCCESS {
			currentMig = nvml.DEVICE_MIG_DISABLE
			if err != nvml.ERROR_NOT_SUPPORTED {
				log.Printf("GetMigMode(%s) error: %v", minor, err)
				c.lastError.WithLabelValues(ordinal, minor, uuid, name, "").Set(float64(err))
				continue
			}
		}

		var numMigs int = 0
		allDevs := []Device{Device{name: name, uuid: uuid, device: dev, instanceId: "", ordinal: ordinal}}
		if currentMig == nvml.DEVICE_MIG_ENABLE {
			numMigs, err = dev.GetMaxMigDeviceCount()
			if err != nvml.SUCCESS {
				log.Printf("GetMaxMigDeviceCount(%s): error: %v", minor, err)
			}
			for j := 0; j < numMigs; j++ {
				migDev, err := dev.GetMigDeviceHandleByIndex(j)
				if err == nvml.SUCCESS {
					migUuid, err := migDev.GetUUID()
					if err != nvml.SUCCESS {
						log.Printf("UUID(minor=%d, mig=%d): error: %v", minorNumber, j, err)
					}
					// Check for GPM
					gpmMig, exists := gpmState[migUuid]
					if !exists {
						if *disableGpm {
							gpmMig = GPM_DISABLED
						} else if gpuQuerySupport, ret := migDev.GpmQueryDeviceSupport(); ret != nvml.SUCCESS {
							log.Printf("GPU(%s) failed to return gpm status, disabling, error: %v", migUuid, ret)
							gpmMig = GPM_DISABLED
						} else {
							if gpuQuerySupport.IsSupportedDevice == 0 {
								gpmMig = GPM_DISABLED
							} else {
								gpmMig = GPM_NO_DATA
							}
							log.Printf("GPU(%s) gpm = %v", migUuid, gpmMig)
						}
						if gpmMig != GPM_DISABLED {
							if allocGpmSamples(migUuid) != nvml.SUCCESS {
								log.Printf("GPU(%s) disabling gpm due to GpmSample alloc errors", migUuid)
								gpmMig = GPM_DISABLED
							}
						}
						gpmState[migUuid] = gpmMig
					}
					migName, err := migDev.GetName()
					if err != nvml.SUCCESS {
						log.Printf("Name(minor=%d, mig=%d): error: %v", minorNumber, j, err)
					}
					migInstanceId, err := migDev.GetGpuInstanceId()
					if err != nvml.SUCCESS {
						log.Printf("Name(minor=%d, mig=%d): error: %v", minorNumber, j, err)
					}
					allDevs = append(allDevs, Device{name: migName, uuid: migUuid, device: migDev, ordinal: ordinal, instanceId: strconv.Itoa(migInstanceId)})
				} else if err != nvml.ERROR_NOT_FOUND {
					log.Printf("GetMigDeviceHandleByInde(%d): error: %v", j, err)
				}
			}
		}

		// Fetch power/temperature/fanspeed first/once so we can reuse it for any MIG cards
		var powerUsageAvailable bool = false
		powerUsage, err := dev.GetPowerUsage()
		if err != nvml.SUCCESS {
			log.Printf("PowerUsage(minor=%s, uuid=%s) error: %v", minor, uuid, err)
			c.lastError.WithLabelValues(ordinal, minor, uuid, name, "").Set(float64(err))
			continue
		} else {
			powerUsageAvailable = true
		}

		var temperatureAvailable bool = false
		temperature, err := dev.GetTemperature(nvml.TEMPERATURE_GPU)
		if err != nvml.SUCCESS {
			log.Printf("Temperature(minor=%s, uuid=%s) error: %v", minor, uuid, err)
			c.lastError.WithLabelValues(ordinal, minor, uuid, name, "").Set(float64(err))
			continue
		} else {
			temperatureAvailable = true
		}

		var fanSpeedAvailable bool = false
		fanSpeed, err := dev.GetFanSpeed()
		if err == nvml.SUCCESS {
			fanSpeedAvailable = true
		}

		for _, err_type := range [2]nvml.MemoryErrorType{nvml.MEMORY_ERROR_TYPE_CORRECTED, nvml.MEMORY_ERROR_TYPE_UNCORRECTED} {
			for _, count_type := range []nvml.EccCounterType{nvml.VOLATILE_ECC, nvml.AGGREGATE_ECC} {
				errCount, err := dev.GetTotalEccErrors(err_type, count_type)
				if err == nvml.SUCCESS {
					c.eccErrors.WithLabelValues(ordinal, minor, uuid, name, eccErrorType[err_type], eccCounterType[count_type]).Set(float64(errCount))
				} else if err != nvml.ERROR_NOT_SUPPORTED {
					log.Printf("GetTotalEccErrors(minor=%s, uuid=%s) error: %v", minor, uuid, err)
				}
			}
		}
		for _, oneDev := range allDevs {
			var jobUid string = ""
			var jobId string = ""
			var slurmInfo string = fmt.Sprintf("/run/gpustat/%s", oneDev.uuid)

			if _, err := os.Stat(slurmInfo); err != nil {
				if oneDev.ordinal != "" {
					if oneDev.instanceId != "" {
						slurmInfo = fmt.Sprintf("/run/gpustat/%s.%s", ordinal, oneDev.instanceId)
					} else {
						slurmInfo = fmt.Sprintf("/run/gpustat/%s", oneDev.ordinal)
					}
					if _, err := os.Stat(slurmInfo); err != nil {
						slurmInfo = ""
					}
				} else {
					slurmInfo = ""
				}
			}

			if slurmInfo != "" {
				content, err := os.ReadFile(slurmInfo)
				if err == nil {
					job_user := strings.Split(strings.TrimSpace(string(content)), " ")
					if len(job_user) <= 2 {
						jobId = job_user[0]
						if len(job_user) == 2 {
							jobUid = job_user[1]
						}
					} else {
						log.Printf("Invalid %s content: %s", slurmInfo, string(content))
					}
					if strings.Contains(string(content), " ") {
						fmt.Sscanf(string(content), "%d %d", &jobId, &jobUid)
					} else {
						fmt.Sscanf(string(content), "%d", &jobId)
					}
				}
			}

			if jobId != "" {
				if f, err := strconv.ParseFloat(jobId, 64); err == nil {
					c.jobId.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(f)
				} else {
					log.Printf("Invalid %s content for jobid: %s (%s)", slurmInfo, jobId, err)
				}
			}
			if jobUid != "" {
				if f, err := strconv.ParseFloat(jobUid, 64); err == nil {
					c.jobUid.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(f)
				} else {
					log.Printf("Invalid %s content for jobuid: %s (%s)", slurmInfo, jobUid, err)
				}
			}

			memory, err := oneDev.device.GetMemoryInfo()
			if err != nvml.SUCCESS {
				log.Printf("MemoryInfo(minor=%s, uuid=%s) error: %v", minor, oneDev.uuid, err)
			} else {
				c.usedMemory.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(float64(memory.Used))
				c.totalMemory.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(float64(memory.Total))
			}

			// GPU cards in MIG mode cannot report Utilization
			if currentMig == nvml.DEVICE_MIG_DISABLE {
				dutyCycle, err := oneDev.device.GetUtilizationRates()
				if err == nvml.SUCCESS {
					c.dutyCycle.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(float64(dutyCycle.Gpu))
				} else if err != nvml.ERROR_NOT_SUPPORTED {
					log.Printf("UtilizationRates(minor=%s, uuid=%s, a=%d) error: %v", minor, oneDev.uuid, nvml.ERROR_NOT_SUPPORTED, err)
				}
			}

			// Common/shared values, set to the same one if available
			if powerUsageAvailable {
				c.powerUsage.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(float64(powerUsage))
			}
			if temperatureAvailable {
				c.temperature.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(float64(temperature))
			}
			if fanSpeedAvailable {
				c.fanSpeed.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(float64(fanSpeed))
			}

			// Collect gpm info
			gpm = gpmState[oneDev.uuid]
			var sample1 nvml.GpmSample = nil
			var sample2 nvml.GpmSample = nil
			if gpm != GPM_DISABLED {
				var ret error
				if (gpm == GPM_NO_DATA) || (gpm == GPM_SAMPLE1_SAMPLE2) {
					ret = oneDev.device.GpmSampleGet(gpmSamples1[oneDev.uuid])
				} else {
					ret = oneDev.device.GpmSampleGet(gpmSamples2[oneDev.uuid])
				}
				if ret != nvml.SUCCESS {
					log.Printf("GPU(%s) error collecting gpm samples: %v", uuid, ret)
					continue
				}
				switch gpm {
				case GPM_NO_DATA:
					gpmState[oneDev.uuid] = GPM_SAMPLE1_ONLY
				case GPM_SAMPLE1_ONLY, GPM_SAMPLE2_SAMPLE1:
					gpmState[oneDev.uuid] = GPM_SAMPLE1_SAMPLE2
					sample1 = gpmSamples1[oneDev.uuid]
					sample2 = gpmSamples2[oneDev.uuid]
				case GPM_SAMPLE1_SAMPLE2:
					gpmState[oneDev.uuid] = GPM_SAMPLE2_SAMPLE1
					sample2 = gpmSamples1[oneDev.uuid]
					sample1 = gpmSamples2[oneDev.uuid]
				}
				if sample1 != nil && sample2 != nil {
					gpmMetric := nvml.GpmMetricsGetType{
						NumMetrics: 13,
						Sample1:    sample1,
						Sample2:    sample2,
						Metrics: [nvml.GPM_METRIC_MAX]nvml.GpmMetric{
							{
								MetricId: uint32(nvml.GPM_METRIC_GRAPHICS_UTIL),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_SM_UTIL),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_SM_OCCUPANCY),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_INTEGER_UTIL),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_ANY_TENSOR_UTIL),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_DRAM_BW_UTIL),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_FP64_UTIL),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_FP32_UTIL),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_FP16_UTIL),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_PCIE_TX_PER_SEC),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_PCIE_RX_PER_SEC),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC),
							},
							{
								MetricId: uint32(nvml.GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC),
							},
						},
					}
					ret = nvml.GpmMetricsGet(&gpmMetric)
					if ret != nvml.SUCCESS {
						log.Printf("GPU(%s) failed to get gpm metrics: %v", oneDev.uuid, ret)
					} else {
						for i := 0; i < int(gpmMetric.NumMetrics); i++ {
							switch int(gpmMetric.Metrics[i].MetricId) {
							case int(nvml.GPM_METRIC_GRAPHICS_UTIL):
								c.graphicsUtil.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value)
							case int(nvml.GPM_METRIC_SM_UTIL):
								c.smUtil.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value)
							case int(nvml.GPM_METRIC_SM_OCCUPANCY):
								c.smOccupancy.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value)
							case int(nvml.GPM_METRIC_INTEGER_UTIL):
								c.integerUtil.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value)
							case int(nvml.GPM_METRIC_ANY_TENSOR_UTIL):
								c.anyTensorUtil.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value)
							case int(nvml.GPM_METRIC_DRAM_BW_UTIL):
								c.dramBwUtil.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value)
							case int(nvml.GPM_METRIC_FP64_UTIL):
								c.fp64Util.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value)
							case int(nvml.GPM_METRIC_FP32_UTIL):
								c.fp32Util.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value)
							case int(nvml.GPM_METRIC_FP16_UTIL):
								c.fp16Util.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value)
							case int(nvml.GPM_METRIC_PCIE_TX_PER_SEC):
								c.pcieTxPerSec.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value * 1024 * 1024)
							case int(nvml.GPM_METRIC_PCIE_RX_PER_SEC):
								c.pcieRxPerSec.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value * 1024 * 1024)
							case int(nvml.GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC):
								c.nvlinkTotalRxPerSec.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value * 1024 * 1024)
							case int(nvml.GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC):
								c.nvlinkTotalTxPerSec.WithLabelValues(ordinal, minor, oneDev.uuid, oneDev.name, oneDev.instanceId, jobId, jobUid).Set(gpmMetric.Metrics[i].Value * 1024 * 1024)
							}
						}
					}
				}
			}
		}
	}
	c.usedMemory.Collect(ch)
	c.totalMemory.Collect(ch)
	c.dutyCycle.Collect(ch)
	c.powerUsage.Collect(ch)
	c.temperature.Collect(ch)
	c.fanSpeed.Collect(ch)
	c.eccErrors.Collect(ch)
	c.lastError.Collect(ch)
	c.jobId.Collect(ch)
	c.jobUid.Collect(ch)
	c.graphicsUtil.Collect(ch)
	c.smUtil.Collect(ch)
	c.smOccupancy.Collect(ch)
	c.integerUtil.Collect(ch)
	c.anyTensorUtil.Collect(ch)
	c.dramBwUtil.Collect(ch)
	c.fp64Util.Collect(ch)
	c.fp32Util.Collect(ch)
	c.fp16Util.Collect(ch)
	c.pcieTxPerSec.Collect(ch)
	c.pcieRxPerSec.Collect(ch)
	c.nvlinkTotalRxPerSec.Collect(ch)
	c.nvlinkTotalTxPerSec.Collect(ch)
}

func metricsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		registry := prometheus.NewRegistry()

		registry.MustRegister(NewCollector())

		gatherers := prometheus.Gatherers{registry}
		if !*disableExporterMetrics {
			gatherers = append(gatherers, prometheus.DefaultGatherer)
		}

		// Delegate http serving to Prometheus client library, which will call collector.Collect.
		h := promhttp.HandlerFor(gatherers, promhttp.HandlerOpts{})
		h.ServeHTTP(w, r)
	}
}

func main() {
	flag.Parse()

	if err := nvml.Init(); err != nvml.SUCCESS {
		log.Fatalf("Couldn't initialize nvml: %v. Make sure NVML is in the shared library search path.", err)
	}
	defer nvml.Shutdown()

	if driverVersion, err := nvml.SystemGetDriverVersion(); err != nvml.SUCCESS {
		log.Printf("SystemGetDriverVersion() error: %v", err)
	} else {
		log.Printf("SystemGetDriverVersion(): %v", driverVersion)
	}

	metricsEndpoint := "/metrics"
	http.Handle(metricsEndpoint, metricsHandler())

	// Serve on all paths under addr
	log.Fatalf("ListenAndServe error: %v", http.ListenAndServe(*addr, nil))
}
