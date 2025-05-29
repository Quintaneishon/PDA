package main

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/mjibson/go-dsp/fft"
	"github.com/xthexder/go-jack"
)

var (
	client      *jack.Client
	inputPort   *jack.Port
	outputPort  *jack.Port
	sampleRate  float64
)

// processCallback is called by JACK in a realtime thread for audio processing
func processCallback(nframes uint32) int {
	// Get input and output buffers
	in := inputPort.GetBuffer(nframes)
	out := outputPort.GetBuffer(nframes)

	// Convert input samples to complex for FFT
	input := make([]complex128, len(in))
	for i, sample := range in {
		input[i] = complex(float64(sample), 0)
	}

	// Perform forward FFT
	spectrum := fft.FFT(input)

	// Here you could modify the frequency domain data
	// For now, we just pass it through

	// Perform inverse FFT
	timeSignal := fft.IFFT(spectrum)

	// Convert back to audio samples and normalize
	for i := range out {
		out[i] = jack.AudioSample(real(timeSignal[i]))
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
