package main

import (
	"fmt"

	"github.com/mjibson/go-dsp/fft"

	//"gonum.org/v1/gonum/cmplxs"
	"image/color"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {
	doas := []float64{-30.0, 40.0} // directional of arrival of both signals in degrees
	d := 20.0                      // distance between microphones in meters
	noiseW := 0.5                  // noise presence (between 0 and 1)

	K := 200.0                   // signal size in samples, also frequency sampling
	w := generateIndices(int(K)) // frequency vector

	///////////////////////////

	freq := []float64{2, 4} // base frecuency for signals

	c := 343.0 // speed of sound

	N := 3 // number of microphones
	r := 2 // number of signals in signal subspace

	// original signals
	s1 := generateSineWave(freq[0], int(K))
	s2 := generateSineWave(freq[1], int(K))

	///////////////////////////
	// Prepare signals for plotting
	signals := []plotter.XYs{complexToXY(s1), complexToXY(s2)}
	labels := []string{"Signal 1", "Signal 2"}

	// Plot and save as PNG
	err := plotSignals(signals, labels, "sine_waves.png")
	if err != nil {
		fmt.Println("Error saving plot:", err)
	}

	///////////////////////////

	x := addComplexSlices(s1, s2)
	y := addComplexSlices(applyPhaseShift(s1, w, d, c, doas[0]), applyPhaseShift(s2, w, d, c, doas[1]))
	z := addComplexSlices(applyPhaseShift(s1, w, 2*d, c, doas[0]), applyPhaseShift(s2, w, 2*d, c, doas[1]))

	addNoise(x, noiseW)
	addNoise(y, noiseW)
	addNoise(z, noiseW)

	///////////////////////////
	// Prepare signals for plotting
	signals = []plotter.XYs{complexToXY(x), complexToXY(y), complexToXY(z)}
	labels = []string{"Signal 1", "Signal 2", "Signal 3"}

	// Plot and save as PNG
	err = plotSignals(signals, labels, "shift_waves.png")
	if err != nil {
		fmt.Println("Error saving plot:", err)
	}

	///////////////////////////
	// Data matrix with noise
	X := computeFFT([][]complex128{x, y, z})

	// Define angles to look for orthogonality
	angles := generateAngles(-90, 90, 0.1) // Generate angles array
	musicSpectrum := make([][]float64, r)
	for i := 0; i < r; i++ {
		musicSpectrum[i] = make([]float64, len(angles))
	}
	// Normally, you should do the next step for each appropriate frequency
	// we're only doing it in the frequencies that most closely fit s1's and s2's frequency

	thisWs := []float64{2, 4}

	for i, thisW := range thisWs {
		thisX := extractColumn(X, thisW, K)
		R := computeOuterProduct(thisX)
		Q, D := eigendecomposition(R)

		/////// Aiuda
		_, indices := sortEigenvalues(D)
		sortedQ := reorderEigenvectors(Q, indices)
		_, Qn := splitEigenvectors(sortedQ, r, N)
		a1 := computeSteeringVectors(N, angles, thisW, d, c)
		spectrum := computeMUSICspectrum(angles, a1, Qn)
		// Extract real values from the spectrum for the current frequency
		for j := 0; j < len(angles); j++ {
			musicSpectrum[i][j] = cmplx.Abs(spectrum.At(j, 0))
		}
	}

	// Print the detected angles for each frequency
	for i, f := range freq {
		maxIdx := argMaxRow(musicSpectrum, i)
		fmt.Printf("Signal with frequency %.0f Hz detected at angle %.2f degrees\n", f, angles[maxIdx])
	}

	// Create and save the MUSIC spectrum plot
	err = plotMUSICSpectrum(angles, musicSpectrum, "music_spectrum.png")
	if err != nil {
		fmt.Printf("Error plotting MUSIC spectrum: %v\n", err)
	} else {
		fmt.Println("MUSIC spectrum plot saved as music_spectrum.png")
	}
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

// Compute steering vectors with delays for different microphones
func computeSteeringVectors(N int, angles []float64, w, d, c float64) *mat.CDense {
	rows, cols := N, len(angles)
	a1 := mat.NewCDense(rows, cols, nil)

	// First microphone (reference, no delay)
	for j := 0; j < cols; j++ {
		a1.Set(0, j, complex(1, 0))
	}

	// Second microphone (delayed by d/c)
	for j, angle := range angles {
		phaseShift := cmplx.Exp(complex(0, -2*math.Pi*w*(d/c)*math.Sin(angle*math.Pi/180)))
		a1.Set(1, j, phaseShift)
	}

	// Third microphone (delayed by 2*d/c)
	for j, angle := range angles {
		phaseShift := cmplx.Exp(complex(0, -2*math.Pi*w*(2*d/c)*math.Sin(angle*math.Pi/180)))
		a1.Set(2, j, phaseShift)
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
