package main

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/runningwild/go-fftw"
	"github.com/xthexder/go-jack"
)

var (
	client      *jack.Client
	inputPort   *jack.Port
	outputPort  *jack.Port
	sampleRate  float64
	iFFT, oFFT  *fftw.DFT1D
)

// processCallback is called by JACK in a realtime thread for audio processing
func processCallback(nframes uint32) int {
	// Get input and output buffers
	in := inputPort.GetBuffer(nframes)
	out := outputPort.GetBuffer(nframes)

	// Convert input samples to complex for FFT
	for i := range in {
		iFFT.In[i] = complex(float64(in[i]), 0)
	}

	// Execute forward FFT
	iFFT.Execute()

	// Copy FFT results - here you could modify the frequency domain data
	copy(oFFT.In, iFFT.Out)

	// Execute inverse FFT
	oFFT.Execute()

	// Convert back to audio samples and normalize
	for i := range out {
		out[i] = jack.AudioSample(real(oFFT.Out[i]) / float64(nframes))
	}

	return 0
}

// shutdownCallback is called when JACK shuts down
func shutdownCallback() {
	fmt.Println("JACK shutdown")
	os.Exit(1)
}

func main() {
	// Open JACK client
	var status int
	client, status = jack.ClientOpen("go_fft", jack.NoStartServer)
	if status != 0 || client == nil {
		fmt.Printf("Failed to connect to JACK: %d\n", status)
		return
	}
	defer client.Close()

	// Set callbacks
	client.SetProcessCallback(processCallback)
	client.OnShutdown(shutdownCallback)

	// Create ports
	inputPort = client.PortRegister("input", jack.DEFAULT_AUDIO_TYPE, jack.PortIsInput, 0)
	outputPort = client.PortRegister("output", jack.DEFAULT_AUDIO_TYPE, jack.PortIsOutput, 0)

	if inputPort == nil || outputPort == nil {
		fmt.Println("Failed to create ports")
		return
	}

	// Get sample rate and buffer size
	sampleRate = float64(client.GetSampleRate())
	bufferSize := client.GetBufferSize()

	fmt.Printf("Sample rate: %v\n", sampleRate)
	fmt.Printf("Buffer size: %v\n", bufferSize)

	// Initialize FFTW
	iFFT = fftw.NewDFT1D(int(bufferSize), fftw.Forward, fftw.OutOfPlace, fftw.Estimate)
	oFFT = fftw.NewDFT1D(int(bufferSize), fftw.Backward, fftw.OutOfPlace, fftw.Estimate)
	defer iFFT.Close()
	defer oFFT.Close()

	// Activate client
	if code := client.Activate(); code != 0 {
		fmt.Printf("Failed to activate client: %d\n", code)
		return
	}

	fmt.Println("Client activated")

	// Connect ports
	ports := client.GetPorts("", "", jack.PortIsPhysical|jack.PortIsOutput)
	if len(ports) > 0 {
		client.Connect(ports[0], inputPort.GetName())
	}

	ports = client.GetPorts("", "", jack.PortIsPhysical|jack.PortIsInput)
	if len(ports) > 0 {
		client.Connect(outputPort.GetName(), ports[0])
	}

	fmt.Println("Ports connected")

	// Wait for signal to quit
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
}
