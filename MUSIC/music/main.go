package main

import (
	"fmt"
	"github.com/scientificgo/fft"
	"math"
	"math/cmplx"
)

func main() {
	// [1]SETUP
	// Direction of arrival of the signal in degrees
	doa := 30.0
	// Distance between microphones in meters
	d := 20.0
	// Noise presence (between 0 and 1)
	noiseW := 0.0
	// Signal size in samples, also frequency sampling
	K := 200.0
	// Frequency vector
	w := append(arange(0, math.Floor(K/2+1), 1), arange(math.Floor(-K/2+1), 0, 1)...)
	////////////////////////////////////////
	// Base frequency of signal of interest (SOI)
	freq := 2
	// Speed of sound
	c := 343.0
	// Time vecto (1 second)
	t := make([]float64, int(K))
	for i, v := range arange(1, K+1, 1) {
		t[i] = v / K
	}

	// Number of signals in signal sub-space
	r := 1
	// Number of microphones
	N := 2

	// [2]
	// Defining the original SOI
	s1 := make([]float64, int(K))
	for i := range s1 {
		s1[i] = math.Sin(2 * math.Pi * float64(freq) * t[i])
	}
	x := make([]float64, int(K))
	copy(x, s1)
	//////////////////////////////
	// Perform FFT
	// res[i] *= cmplx.Exp(complex(-1, 0) * complex(2*math.Pi*w[i]*(d/c)*math.Sin(doa*math.Pi/180), 0))

	complexX := floatArr2ComplexArr(x)
	res := fft.Fft(complexX, false)

	fmt.Println(complexX[0])
	fmt.Println(complexX[len(complexX)-1])

	fmt.Println(res[0])
	fmt.Println(res[len(res)-1])

	for i := range s1 {
		res[i] *= res[i] * cmplx.Exp(complex(-1, 0)*complex(2*math.Pi*w[i]*(d/c)*math.Sin(doa*math.Pi/180), 0))
	}

	//yComplex := fft.IFFT(res)
	y := make([]float64, int(K))
	//for i := range yComplex {
	//	y[i] = real(yComplex[i])
	//}
	fmt.Println(res)
	fmt.Println(y)
	fmt.Println(x)
	fmt.Println(w)
	fmt.Println(t)
	fmt.Println(doa, d, noiseW, freq, c, r, N)
}

func arange(start, stop, step float64) []float64 {
	N := int(math.Ceil((stop - start) / step))
	rnge := make([]float64, N)
	i := 0
	for x := start; x < stop; x += step {
		rnge[i] = x
		i += 1
	}
	return rnge
}

func floatArr2ComplexArr(arr []float64) []complex128 {
	complexArr := make([]complex128, len(arr))
	for i, v := range arr {
		complexArr[i] = complex(v, 0)
	}
	return complexArr
}
