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
	jobId       *prometheus.GaugeVec
	jobUid      *prometheus.GaugeVec
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

	numDevices, err := nvml.DeviceGetCount()
	if err != nvml.SUCCESS {
		log.Printf("DeviceCount() error: %v", err)
		return
	} else {
		c.numDevices.Set(float64(numDevices))
		ch <- c.numDevices
	}

	for i := 0; i < int(numDevices); i++ {
		dev, err := nvml.DeviceGetHandleByIndex(i)
		if err != nvml.SUCCESS {
			log.Printf("DeviceHandleByIndex(%d) error: %v", i, err)
			continue
		}

		minorNumber, err := dev.GetMinorNumber()
		if err != nvml.SUCCESS {
			log.Printf("MinorNumber() error: %v", err)
			continue
		}
		minor := strconv.Itoa(int(minorNumber))

		uuid, err := dev.GetUUID()
		if err != nvml.SUCCESS {
			log.Printf("UUID() error: %v", err)
			continue
		}

		name, err := dev.GetName()
		if err != nvml.SUCCESS {
			log.Printf("Name() error: %v", err)
			continue
		}

		memory, err := dev.GetMemoryInfo()
		if err != nvml.SUCCESS {
			log.Printf("MemoryInfo() error: %v", err)
		} else {
			c.usedMemory.WithLabelValues(minor, uuid, name).Set(float64(memory.Used))
			c.totalMemory.WithLabelValues(minor, uuid, name).Set(float64(memory.Total))
		}

		dutyCycle, err := dev.GetUtilizationRates()
		if err != nvml.SUCCESS {
			log.Printf("UtilizationRates() error: %v", err)
		} else {
			c.dutyCycle.WithLabelValues(minor, uuid, name).Set(float64(dutyCycle.Gpu))
		}

		powerUsage, err := dev.GetPowerUsage()
		if err != nvml.SUCCESS {
			log.Printf("PowerUsage() error: %v", err)
		} else {
			c.powerUsage.WithLabelValues(minor, uuid, name).Set(float64(powerUsage))
		}

		temperature, err := dev.GetTemperature(nvml.TEMPERATURE_GPU)
		if err != nvml.SUCCESS {
			log.Printf("Temperature() error: %v", err)
		} else {
			c.temperature.WithLabelValues(minor, uuid, name).Set(float64(temperature))
		}

		fanSpeed, err := dev.GetFanSpeed()
		if err == nvml.SUCCESS {
			c.fanSpeed.WithLabelValues(minor, uuid, name).Set(float64(fanSpeed))
		}

		var jobUid int64 = 0
		var jobId int64 = 0
		var slurmInfo string = fmt.Sprintf("/run/gpustat/%d", i)

		if _, err := os.Stat(slurmInfo); err == nil {
			content, err := ioutil.ReadFile(slurmInfo)
			if err == nil {
				fmt.Sscanf(string(content), "%d %d", &jobId, &jobUid)
			}
		}

		c.jobId.WithLabelValues(minor, uuid, name).Set(float64(jobId))
		c.jobUid.WithLabelValues(minor, uuid, name).Set(float64(jobUid))
	}
	c.usedMemory.Collect(ch)
	c.totalMemory.Collect(ch)
	c.dutyCycle.Collect(ch)
	c.powerUsage.Collect(ch)
	c.temperature.Collect(ch)
	c.fanSpeed.Collect(ch)
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
