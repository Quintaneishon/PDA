package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"os/signal"
	"sort"
	"syscall"

	"github.com/mjibson/go-dsp/fft"
	"github.com/xthexder/go-jack"
	"gonum.org/v1/gonum/mat"
)

var (
	client      *jack.Client
	inputPorts  []*jack.Port
	outputPort  *jack.Port
	sampleRate  float64
	bufferSize  uint32
	N           = 3     // number of microphones
	r           = 2     // number of signals in signal subspace
	d           = 0.18  // distance between microphones in meters (18cm)
	c           = 343.0 // speed of sound

	// Triangular array geometry (angles in degrees)
	micAngles = []float64{0, 120, 240} // Angles of microphones in the triangle
	
	// Voice detection parameters
	energyThreshold = 0.01  // Adjust this value based on your setup
	minFreq        = 85.0   // Minimum frequency to analyze (Hz)
	maxFreq        = 255.0  // Maximum frequency to analyze (Hz)
	lastAngle      = 0.0    // Store last valid angle
)

// Calculate signal energy in the voice frequency range
func calculateVoiceEnergy(spectrum []complex128, sampleRate float64) float64 {
	binSize := sampleRate / float64(len(spectrum))
	minBin := int(minFreq / binSize)
	maxBin := int(maxFreq / binSize)
	
	if maxBin >= len(spectrum) {
		maxBin = len(spectrum) - 1
	}

	energy := 0.0
	for i := minBin; i <= maxBin; i++ {
		energy += cmplx.Abs(spectrum[i]) * cmplx.Abs(spectrum[i])
	}
	return energy / float64(maxBin-minBin+1)
}

// processCallback is called by JACK in a realtime thread for audio processing
func processCallback(nframes uint32) int {
	// Get input buffers from all microphones
	inputs := make([][]jack.AudioSample, N)
	for i := 0; i < N; i++ {
		inputs[i] = inputPorts[i].GetBuffer(nframes)
	}

	// Convert input samples to complex for FFT
	complexInputs := make([][]complex128, N)
	for i := 0; i < N; i++ {
		complexInputs[i] = make([]complex128, len(inputs[i]))
		for j, sample := range inputs[i] {
			complexInputs[i][j] = complex(float64(sample), 0)
		}
	}

	// Perform FFT on each input
	X := make([][]complex128, N)
	for i := 0; i < N; i++ {
		X[i] = fft.FFT(complexInputs[i])
	}

	// Check voice energy in first microphone
	energy := calculateVoiceEnergy(X[0], sampleRate)
	
	if energy > energyThreshold {
		// Process voice frequency range
		binSize := sampleRate / float64(len(X[0]))
		freqBins := make([]int, 0)
		
		// Get all bins in voice frequency range
		for freq := minFreq; freq <= maxFreq; freq += binSize {
			bin := int(freq * float64(len(X[0])) / sampleRate)
			if bin < len(X[0]) {
				freqBins = append(freqBins, bin)
			}
		}

		// Average DOA estimation over voice frequency range
		var sumAngle float64
		var maxPower float64
		validEstimates := 0

		for _, freqBin := range freqBins {
			// Extract the frequency bin of interest
			thisX := make([]complex128, N)
			for i := 0; i < N; i++ {
				thisX[i] = X[i][freqBin]
			}

			// Compute correlation matrix
			R := computeOuterProduct(thisX)

			// Perform eigendecomposition
			Q, D := eigendecomposition(R)
			_, indices := sortEigenvalues(D)
			sortedQ := reorderEigenvectors(Q, indices)
			_, Qn := splitEigenvectors(sortedQ, r, N)

			// Generate angles and compute MUSIC spectrum
			angles := generateAngles(-90, 90, 1)
			a1 := computeSteeringVectors(N, angles, float64(freqBin)*binSize, d, c)
			spectrum := computeMUSICspectrum(angles, a1, Qn)

			// Find peak in spectrum
			maxVal := 0.0
			maxIdx := 0
			for i := 0; i < len(angles); i++ {
				val := cmplx.Abs(spectrum.At(i, 0))
				if val > maxVal {
					maxVal = val
					maxIdx = i
				}
			}

			// Weight the angle by its power
			power := cmplx.Abs(spectrum.At(maxIdx, 0))
			if power > 0.1 { // Threshold for valid estimation
				sumAngle += angles[maxIdx] * power
				maxPower += power
				validEstimates++
			}
		}

		// Update angle if we have valid estimates
		if validEstimates > 0 {
			lastAngle = sumAngle / maxPower
			fmt.Printf("\rVoice detected! Estimated DOA: %.1f degrees (Energy: %.6f)", lastAngle, energy)
		}
	} else {
		fmt.Printf("\rNo voice detected (Energy: %.6f)                              ", energy)
	}

	// Pass through audio to output (using first input)
	out := outputPort.GetBuffer(nframes)
	copy(out, inputs[0])

	return 0
}

// shutdownCallback is called when JACK shuts down
func shutdownCallback() {
	fmt.Println("\nJACK shutdown")
	os.Exit(1)
}

func main() {
	// Open JACK client
	var status int
	client, status = jack.ClientOpen("doa_estimator", jack.NoStartServer)
	if status != 0 || client == nil {
		fmt.Printf("Failed to connect to JACK: %d\n", status)
		return
	}
	defer client.Close()

	// Set callbacks
	client.SetProcessCallback(processCallback)
	client.OnShutdown(shutdownCallback)

	// Create input ports for each microphone
	inputPorts = make([]*jack.Port, N)
	for i := 0; i < N; i++ {
		portName := fmt.Sprintf("input_%d", i+1)
		inputPorts[i] = client.PortRegister(portName, jack.DEFAULT_AUDIO_TYPE, jack.PortIsInput, 0)
		if inputPorts[i] == nil {
			fmt.Printf("Failed to create input port %d\n", i+1)
			return
		}
	}

	// Create output port (for monitoring)
	outputPort = client.PortRegister("output", jack.DEFAULT_AUDIO_TYPE, jack.PortIsOutput, 0)
	if outputPort == nil {
		fmt.Println("Failed to create output port")
		return
	}

	// Get sample rate and buffer size
	sampleRate = float64(client.GetSampleRate())
	bufferSize = client.GetBufferSize()

	fmt.Printf("Sample rate: %v Hz\n", sampleRate)
	fmt.Printf("Buffer size: %v samples\n", bufferSize)
	fmt.Printf("Voice frequency range: %.0f-%.0f Hz\n", minFreq, maxFreq)
	fmt.Printf("Energy threshold: %.6f\n", energyThreshold)

	// Activate client
	if code := client.Activate(); code != 0 {
		fmt.Printf("Failed to activate client: %d\n", code)
		return
	}

	fmt.Println("Client activated")
	fmt.Println("Connect your microphone inputs to the input_1, input_2, and input_3 ports")
	fmt.Println("Press Ctrl+C to exit")

	// Wait for signal to quit
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
}

// Find the index of max value in a row
func argMaxRow(matrix [][]float64, row int) int {
	maxIdx := 0
	maxVal := matrix[row][0]

	for j := 1; j < len(matrix[row]); j++ {
		if matrix[row][j] > maxVal {
			maxVal = matrix[row][j]
			maxIdx = j
		}
	}

	return maxIdx
}

// Perform complex matrix multiplication
func complexMatrixMul(A, B *mat.CDense) *mat.CDense {
	rowsA, colsA := A.Dims()
	rowsB, colsB := B.Dims()

	if colsA != rowsB {
		panic("Invalid matrix dimensions for multiplication")
	}

	C := mat.NewCDense(rowsA, colsB, nil)

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			sum := complex(0, 0)
			for k := 0; k < colsA; k++ {
				sum += A.At(i, k) * B.At(k, j)
			}
			C.Set(i, j, sum)
		}
	}

	return C
}

// Compute MUSIC spectrum manually
func computeMUSICspectrum(angles []float64, a1, Qn *mat.CDense) *mat.CDense {
	rows, _ := a1.Dims()
	musicSpectrum := mat.NewCDense(len(angles), 1, nil)

	QnH := conjugateTranspose(Qn)
	term1 := complexMatrixMul(Qn, QnH)

	for k := range angles {
		aVec := mat.NewCDense(rows, 1, nil)
		for i := 0; i < rows; i++ {
			aVec.Set(i, 0, a1.At(i, k))
		}

		aVecH := conjugateTranspose(aVec)
		term2 := complexMatrixMul(aVecH, term1)
		term3 := complexMatrixMul(term2, aVec)

		musicSpectrum.Set(k, 0, complex(1, 0)/term3.At(0, 0))
	}

	return musicSpectrum
}

// Compute steering vectors with delays for triangular microphone array
func computeSteeringVectors(N int, angles []float64, w, d, c float64) *mat.CDense {
	rows, cols := N, len(angles)
	a1 := mat.NewCDense(rows, cols, nil)

	// First microphone (reference, no delay)
	for j := 0; j < cols; j++ {
		a1.Set(0, j, complex(1, 0))
	}

	// For each microphone after the reference
	for i := 1; i < N; i++ {
		micAngleRad := micAngles[i] * math.Pi / 180.0
		
		for j, angle := range angles {
			angleRad := angle * math.Pi / 180.0
			delay := -d * math.Cos(micAngleRad - angleRad)
			phaseShift := cmplx.Exp(complex(0, -2*math.Pi*w*(delay/c)))
			a1.Set(i, j, phaseShift)
		}
	}

	return a1
}

// Extract signal and noise eigenvectors
func splitEigenvectors(Q *mat.CDense, r int, N int) (signal *mat.CDense, noise *mat.CDense) {
	rows, cols := Q.Dims()

	if r < 0 || r > N || N > cols {
		panic("Invalid values for r or N")
	}

	signal = mat.NewCDense(rows, r, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < rows; j++ {
			signal.Set(j, i, Q.At(j, i))
		}
	}

	noiseCols := N - r
	noise = mat.NewCDense(rows, noiseCols, nil)
	for i := r; i < N; i++ {
		for j := 0; j < rows; j++ {
			noise.Set(j, i-r, Q.At(j, i))
		}
	}

	return signal, noise
}

// DiagonalElement stores a complex number and its original index
type DiagonalElement struct {
	Value float64
	Index int
}

// Sort eigenvalues in descending order and return sorted values and indices
func sortEigenvalues(eigenvalues []complex128) ([]float64, []int) {
	diagElements := make([]DiagonalElement, len(eigenvalues))
	for i, val := range eigenvalues {
		diagElements[i] = DiagonalElement{
			Value: cmplx.Abs(val),
			Index: i,
		}
	}

	sort.Slice(diagElements, func(i, j int) bool {
		return diagElements[i].Value > diagElements[j].Value
	})

	sortedValues := make([]float64, len(eigenvalues))
	indices := make([]int, len(eigenvalues))
	for i, elem := range diagElements {
		sortedValues[i] = elem.Value
		indices[i] = elem.Index
	}

	return sortedValues, indices
}

// Compute eigendecomposition of a matrix R
func eigendecomposition(R [][]complex128) (*mat.CDense, []complex128) {
	n := len(R)
	cDense := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			cDense.Set(i, j, real(R[i][j]))
		}
	}

	var eig mat.Eigen
	ok := eig.Factorize(cDense, mat.EigenRight)
	if !ok {
		panic("Eigendecomposition failed")
	}
	eigenvalues := eig.Values(nil)
	eigenvectors := mat.NewCDense(n, n, nil)
	eig.VectorsTo(eigenvectors)

	return eigenvectors, eigenvalues
}

// Compute conjugate transpose manually
func conjugateTranspose(X *mat.CDense) *mat.CDense {
	rows, cols := X.Dims()
	XT := mat.NewCDense(cols, rows, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			XT.Set(j, i, cmplx.Conj(X.At(i, j)))
		}
	}
	return XT
}

// Function to compute outer product (R = thisX * thisX)
func computeOuterProduct(X []complex128) [][]complex128 {
	n := len(X)
	R := make([][]complex128, n)

	for i := 0; i < n; i++ {
		R[i] = make([]complex128, n)
		for j := 0; j < n; j++ {
			R[i][j] = X[i] * cmplx.Conj(X[j])
		}
	}

	return R
}

// Generate angle range similar to np.arange(-90, 90, 0.1)
func generateAngles(start, stop, step float64) []float64 {
	var angles []float64
	for angle := start; angle < stop; angle += step {
		angles = append(angles, angle)
	}
	return angles
}

// Reorder eigenvectors according to sorted indices
func reorderEigenvectors(Q *mat.CDense, indices []int) *mat.CDense {
	rows, cols := Q.Dims()
	newQ := mat.NewCDense(rows, cols, nil)

	for newPos, oldPos := range indices {
		for i := 0; i < rows; i++ {
			newQ.Set(i, newPos, Q.At(i, oldPos))
		}
	}

	return newQ
}
