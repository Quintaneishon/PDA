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

// Function to plot MUSIC spectrum
func plotMUSICSpectrum(angles []float64, musicSpectrum [][]float64, filename string) error {
	p := plot.New()
	p.Title.Text = "MUSIC"
	p.X.Label.Text = "Angles (deg)"
	p.Y.Label.Text = "Spectrum"
	p.Legend.Top = true
	p.Add(plotter.NewGrid())

	// Plot first dataset (2Hz, hotpink)
	points1 := make(plotter.XYs, len(angles))
	for i, angle := range angles {
		points1[i].X = angle
		points1[i].Y = musicSpectrum[0][i]
	}
	line1, err := plotter.NewLine(points1)
	if err != nil {
		return err
	}
	line1.Color = color.RGBA{R: 255, G: 105, B: 180, A: 255} // Hotpink
	p.Add(line1)
	p.Legend.Add("2Hz", line1)

	// Plot second dataset (4Hz, steelblue)
	points2 := make(plotter.XYs, len(angles))
	for i, angle := range angles {
		points2[i].X = angle
		points2[i].Y = musicSpectrum[1][i]
	}
	line2, err := plotter.NewLine(points2)
	if err != nil {
		return err
	}
	line2.Color = color.RGBA{R: 70, G: 130, B: 180, A: 255} // Steelblue
	p.Add(line2)
	p.Legend.Add("4Hz", line2)

	return p.Save(6*vg.Inch, 4*vg.Inch, filename)
}

// Perform complex matrix multiplication
func complexMatrixMul(A, B *mat.CDense) *mat.CDense {
	rowsA, colsA := A.Dims()
	rowsB, colsB := B.Dims()

	if colsA != rowsB {
		panic("Invalid matrix dimensions for multiplication")
	}

	C := mat.NewCDense(rowsA, colsB, nil) // Resulting matrix

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			sum := complex(0, 0)
			for k := 0; k < colsA; k++ {
				sum += A.At(i, k) * B.At(k, j) // Dot product sum
			}
			C.Set(i, j, sum)
		}
	}

	return C
}

// Compute MUSIC spectrum manually
func computeMUSICspectrum(angles []float64, a1, Qn *mat.CDense) *mat.CDense {
	rows, _ := a1.Dims() // Get dimensions from steering vectors matrix
	musicSpectrum := mat.NewCDense(len(angles), 1, nil)

	// Compute Qn^H (Hermitian transpose)
	QnH := conjugateTranspose(Qn)

	// Compute Qn @ Qn^H using custom complex multiplication
	term1 := complexMatrixMul(Qn, QnH)

	// Compute spectrum
	for k := range angles {
		// Extract column vector a1[:, k]
		aVec := mat.NewCDense(rows, 1, nil)
		for i := 0; i < rows; i++ {
			aVec.Set(i, 0, a1.At(i, k))
		}

		// Compute a1^H @ term1 @ a1 manually
		aVecH := conjugateTranspose(aVec)
		term2 := complexMatrixMul(aVecH, term1)
		term3 := complexMatrixMul(term2, aVec)

		// Compute MUSIC spectrum value
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
		micAngleRad := micAngles[i] * math.Pi / 180.0 // Convert mic angle to radians
		
		for j, angle := range angles {
			angleRad := angle * math.Pi / 180.0 // Convert incoming sound angle to radians
			
			// Calculate delay using the formula: −dm3∗cos(θm−θ)
			delay := -d * math.Cos(micAngleRad - angleRad)
			
			// Calculate phase shift
			phaseShift := cmplx.Exp(complex(0, -2*math.Pi*w*(delay/c)))
			a1.Set(i, j, phaseShift)
		}
	}

	return a1
}

// Extract signal and noise eigenvectors
func splitEigenvectors(Q *mat.CDense, r int, N int) (signal *mat.CDense, noise *mat.CDense) {
	rows, cols := Q.Dims()

	// Ensure r and N are within the valid range
	if r < 0 || r > N || N > cols {
		panic("Invalid values for r or N")
	}

	// Signal eigenvectors (first r columns)
	signal = mat.NewCDense(rows, r, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < rows; j++ {
			signal.Set(j, i, Q.At(j, i))
		}
	}

	// Noise eigenvectors (columns from r to N)
	noiseCols := N - r
	noise = mat.NewCDense(rows, noiseCols, nil)
	for i := r; i < N; i++ {
		for j := 0; j < rows; j++ {
			noise.Set(j, i-r, Q.At(j, i)) // Adjust column index
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
	// Create a slice of DiagonalElement to store values and indices
	diagElements := make([]DiagonalElement, len(eigenvalues))
	for i, val := range eigenvalues {
		diagElements[i] = DiagonalElement{
			Value: cmplx.Abs(val), // Use magnitude for sorting complex numbers
			Index: i,
		}
	}

	// Sort by magnitude in descending order
	sort.Slice(diagElements, func(i, j int) bool {
		return diagElements[i].Value > diagElements[j].Value
	})

	// Extract sorted values and indices
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
	// Convert [][]complex128 to mat.CDense
	n := len(R)
	cDense := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			cDense.Set(i, j, real(R[i][j]))
		}
	}

	// Compute eigenvalues and eigenvectors
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
	XT := mat.NewCDense(cols, rows, nil) // Transposed matrix

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			XT.Set(j, i, cmplx.Conj(X.At(i, j))) // Swap indices & conjugate
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
			R[i][j] = X[i] * cmplx.Conj(X[j]) // Multiply with conjugate
		}
	}

	return R
}

// Function to extract a column from [][]complex128 for a given frequency
func extractColumn(X [][]complex128, freq float64, K float64) []complex128 {
	N := len(X[0]) // FFT size

	freqIndex := int(math.Round(freq * float64(N) / K))

	// Ensure the index is within bounds
	if freqIndex >= N {
		freqIndex = N - 1
	} else if freqIndex < 0 {
		freqIndex = 0
	}

	var column []complex128
	for i := 0; i < len(X); i++ {
		column = append(column, X[i][freqIndex])
	}
	return column
}

func generateIndices(K int) []float64 {
	w := make([]float64, K)
	halfK := K / 2

	// First half: [0, 1, ..., K/2]
	for i := 0; i <= halfK; i++ {
		w[i] = float64(i)
	}

	// Second half: [-(K/2)+1, ..., -1]
	for i := halfK + 1; i < K; i++ {
		w[i] = float64(i - K)
	}

	return w
}

func generateSineWave(freq float64, K int) []complex128 {
	sineWave := make([]complex128, K)

	// Generate sine wave with complex representation
	for i := 0; i < K; i++ {
		t := float64(i+1) / float64(K)                       // Time vector
		sineWave[i] = complex(math.Sin(2*math.Pi*freq*t), 0) // Convert to complex128
	}

	return sineWave
}

func addComplexSlices(a, b []complex128) []complex128 {
	if len(a) != len(b) {
		fmt.Println("Error: Arrays must be of equal length!")
		return nil
	}

	result := make([]complex128, len(a))
	for i := range a {
		result[i] = a[i] + b[i] // Element-wise sum
	}

	return result
}

func applyPhaseShift(input []complex128, w []float64, d, c float64, doa float64) []complex128 {
	transformed := fft.FFT(input)

	// Apply phase shift
	for i := range transformed {
		phaseShift := cmplx.Exp(complex(0, -2*math.Pi*w[i]*(d/c)*math.Sin(doa*math.Pi/180)))
		transformed[i] *= phaseShift
	}

	// Compute IFFT
	return fft.IFFT(transformed)
}

// Compute FFT for multiple signals and store in a 2D slice
func computeFFT(signals [][]complex128) [][]complex128 {
	fftResults := make([][]complex128, len(signals))

	for i, signal := range signals {
		fftResults[i] = fft.FFT(signal) // Compute FFT for each signal
	}

	return fftResults
}

// Apply Gaussian noise to a complex array
func addNoise(x []complex128, noiseW float64) {
	src := rand.NewSource(time.Now().UnixNano())
	rng := rand.New(src) // New random number generator

	for i := range x {
		noise := rng.Float64() * (noiseW / 10) // Generate Gaussian noise
		x[i] += complex(noise, 0)              // Apply noise to the real part
	}
}

// Convert complex signals to a plot-friendly format
func complexToXY(signal []complex128) plotter.XYs {
	points := make(plotter.XYs, len(signal))
	for i, val := range signal {
		points[i].X = float64(i) // Time index
		points[i].Y = real(val)  // Extract the real part
	}
	return points
}

// External function for plotting
func plotSignals(signals []plotter.XYs, labels []string, filename string) error {
	p := plot.New()
	p.Title.Text = "Sine Waves"
	p.X.Label.Text = "Time"
	p.Y.Label.Text = "Amplitude"

	for i, sig := range signals {
		line, err := plotter.NewLine(sig)
		if err != nil {
			return fmt.Errorf("error creating plot line: %v", err)
		}
		line.Color = plotutil.Color(i)
		p.Add(line)
		p.Legend.Add(labels[i], line)
	}

	// Save plot
	return p.Save(8*vg.Inch, 4*vg.Inch, filename)
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

	// Copy columns to new positions based on indices
	for newPos, oldPos := range indices {
		for i := 0; i < rows; i++ {
			newQ.Set(i, newPos, Q.At(i, oldPos))
		}
	}

	return newQ
}
