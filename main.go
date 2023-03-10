package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	namespace = "nvidia_gpu"
)

var (
	addr = flag.String("web.listen-address", ":9445", "Address to listen on for web interface and telemetry.")

	labels = []string{"minor_number", "uuid", "name"}
)

type Collector struct {
	sync.Mutex
	numDevices  prometheus.Gauge
	usedMemory  *prometheus.GaugeVec
	totalMemory *prometheus.GaugeVec
	dutyCycle   *prometheus.GaugeVec
	powerUsage  *prometheus.GaugeVec
	temperature *prometheus.GaugeVec
	fanSpeed    *prometheus.GaugeVec
	lastError   *prometheus.GaugeVec
	jobId       *prometheus.GaugeVec
	jobUid      *prometheus.GaugeVec
}

type Device struct {
	name, uuid, slurmName string
	isMig                 bool
	device                nvml.Device
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
			labels,
		),
		totalMemory: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "memory_total_bytes",
				Help:      "Total memory of the GPU device in bytes",
			},
			labels,
		),
		dutyCycle: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "duty_cycle",
				Help:      "Percent of time over the past sample period during which one or more kernels were executing on the GPU device",
			},
			labels,
		),
		powerUsage: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "power_usage_milliwatts",
				Help:      "Power usage of the GPU device in milliwatts",
			},
			labels,
		),
		temperature: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "temperature_celsius",
				Help:      "Temperature of the GPU device in celsius",
			},
			labels,
		),
		fanSpeed: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "fanspeed_percent",
				Help:      "Fanspeed of the GPU device as a percent of its maximum",
			},
			labels,
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
			labels,
		),
		jobUid: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "jobUid",
				Help:      "Uid number of user running jobs on this GPU",
			},
			labels,
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

		dev, err := nvml.DeviceGetHandleByIndex(i)
		if err != nvml.SUCCESS {
			log.Printf("DeviceHandleByIndex(%d) error: %v", i, err)
			continue
		}

		minorNumber, err := dev.GetMinorNumber()
		if err != nvml.SUCCESS {
			log.Printf("MinorNumber(%d) error: %v", i, err)
			c.lastError.WithLabelValues(strconv.Itoa(i), "", "").Set(float64(err))
			continue
		}
		minor := strconv.Itoa(int(minorNumber))

		uuid, err := dev.GetUUID()
		if err != nvml.SUCCESS {
			log.Printf("UUID(%s) error: %v", minor, err)
			c.lastError.WithLabelValues(minor, "", "").Set(float64(err))
			continue
		}

		name, err := dev.GetName()
		if err != nvml.SUCCESS {
			log.Printf("Name(%s) error: %v", minor, err)
			c.lastError.WithLabelValues(minor, uuid, "").Set(float64(err))
			continue
		}

		currentMig, _, err := dev.GetMigMode()
		if err != nvml.SUCCESS {
			currentMig = nvml.DEVICE_MIG_DISABLE
			if err != nvml.ERROR_NOT_SUPPORTED {
				log.Printf("GetMigMode(%s) error: %v", minor, err)
				c.lastError.WithLabelValues(minor, uuid, name).Set(float64(err))
				continue
			}
		}

		var numMigs int = 0
		allDevs := []Device{Device{name: name, uuid: uuid, device: dev, isMig: false, slurmName: minor}}
		if currentMig == nvml.DEVICE_MIG_ENABLE {
			numMigs, err = dev.GetMaxMigDeviceCount()
			if err != nvml.SUCCESS {
				log.Printf("GetMaxMigDeviceCount(%s): error: %v", minor, err)
			}
			for j := 0; j < numMigs; j++ {
				migDev, err := dev.GetMigDeviceHandleByIndex(j)
				if err != nvml.SUCCESS {
					log.Printf("GetMigDeviceHandleByInde(%d): error: %v", j, err)
				} else {
					migUuid, err := migDev.GetUUID()
					if err != nvml.SUCCESS {
						log.Printf("UUID(minor=%d, mig=%d): error: %v", minorNumber, j, err)
					}
					migName, err := migDev.GetName()
					if err != nvml.SUCCESS {
						log.Printf("Name(minor=%d, mig=%d): error: %v", minorNumber, j, err)
					}
					allDevs = append(allDevs, Device{name: migName, uuid: migUuid, device: migDev, isMig: true, slurmName: migUuid})
				}
			}
		}

		// check, just in case
		if (numMigs + 1) != len(allDevs) {
			log.Printf("MIG: found %d devices but was expecting %d", len(allDevs)-1, numMigs)
		}

		// Fetch power/temperature/fanspeed first/once so we can reuse it for any MIG cards
		var powerUsageAvailable bool = false
		powerUsage, err := dev.GetPowerUsage()
		if err != nvml.SUCCESS {
			log.Printf("PowerUsage(minor=%s, uuid=%s) error: %v", minor, uuid, err)
			c.lastError.WithLabelValues(minor, uuid, name).Set(float64(err))
			continue
		} else {
			powerUsageAvailable = true
		}

		var temperatureAvailable bool = false
		temperature, err := dev.GetTemperature(nvml.TEMPERATURE_GPU)
		if err != nvml.SUCCESS {
			log.Printf("Temperature(minor=%s, uuid=%s) error: %v", minor, uuid, err)
			c.lastError.WithLabelValues(minor, uuid, name).Set(float64(err))
			continue
		} else {
			temperatureAvailable = true
		}

		var fanSpeedAvailable bool = false
		fanSpeed, err := dev.GetFanSpeed()
		if err == nvml.SUCCESS {
			fanSpeedAvailable = true
		}
		for _, oneDev := range allDevs {
			memory, err := oneDev.device.GetMemoryInfo()
			if err != nvml.SUCCESS {
				log.Printf("MemoryInfo(minor=%s, uuid=%s) error: %v", minor, oneDev.uuid, err)
			} else {
				c.usedMemory.WithLabelValues(minor, oneDev.uuid, oneDev.name).Set(float64(memory.Used))
				c.totalMemory.WithLabelValues(minor, oneDev.uuid, oneDev.name).Set(float64(memory.Total))
			}

			// GPU cards in MIG mode cannot report Utilization
			if currentMig == nvml.DEVICE_MIG_DISABLE {
				dutyCycle, err := oneDev.device.GetUtilizationRates()
				if err == nvml.SUCCESS {
					c.dutyCycle.WithLabelValues(minor, oneDev.uuid, oneDev.name).Set(float64(dutyCycle.Gpu))
				} else if err != nvml.ERROR_NOT_SUPPORTED {
					log.Printf("UtilizationRates(minor=%s, uuid=%s, a=%d) error: %v", minor, oneDev.uuid, nvml.ERROR_NOT_SUPPORTED, err)
				}
			}

			// Common/shared values, set to the same one if available
			if powerUsageAvailable {
				c.powerUsage.WithLabelValues(minor, oneDev.uuid, oneDev.name).Set(float64(powerUsage))
			}
			if temperatureAvailable {
				c.temperature.WithLabelValues(minor, oneDev.uuid, oneDev.name).Set(float64(temperature))
			}
			if fanSpeedAvailable {
				c.fanSpeed.WithLabelValues(minor, oneDev.uuid, oneDev.name).Set(float64(fanSpeed))
			}

			var jobUid int64 = 0
			var jobId int64 = 0
			var slurmInfo string = fmt.Sprintf("/run/gpustat/%s", oneDev.slurmName)

			if _, err := os.Stat(slurmInfo); err == nil {
				content, err := ioutil.ReadFile(slurmInfo)
				if err == nil {
					fmt.Sscanf(string(content), "%d %d", &jobId, &jobUid)
				}
			}

			c.jobId.WithLabelValues(minor, oneDev.uuid, oneDev.name).Set(float64(jobId))
			c.jobUid.WithLabelValues(minor, oneDev.uuid, oneDev.name).Set(float64(jobUid))
		}
	}
	c.usedMemory.Collect(ch)
	c.totalMemory.Collect(ch)
	c.dutyCycle.Collect(ch)
	c.powerUsage.Collect(ch)
	c.temperature.Collect(ch)
	c.fanSpeed.Collect(ch)
	c.lastError.Collect(ch)
	c.jobId.Collect(ch)
	c.jobUid.Collect(ch)
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

	prometheus.MustRegister(NewCollector())

	// Serve on all paths under addr
	log.Fatalf("ListenAndServe error: %v", http.ListenAndServe(*addr, promhttp.Handler()))
}
