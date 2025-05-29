# JACK FFT Audio Processor

This is a Go implementation of a JACK client that performs real-time FFT processing on audio signals. It takes audio input, performs FFT analysis, and outputs the processed audio.

## Prerequisites

1. JACK Audio Connection Kit
2. FFTW3 library
3. Go 1.23 or later

### Installing Dependencies

On macOS:
```bash
brew install jack fftw
```

On Ubuntu/Debian:
```bash
sudo apt-get install libjack-jackd2-dev libfftw3-dev
```

## Building

```bash
go mod tidy
go build
```

## Running

1. Start the JACK server (if not already running)
2. Run the program:
```bash
./music_go_aira
```

The program will create:
- An input port named "input"
- An output port named "output"

It will automatically connect to the first available physical input and output ports.

## Usage

The program performs:
1. Forward FFT on input audio
2. Allows for frequency domain processing (currently just passes through)
3. Inverse FFT to convert back to time domain

To stop the program, press Ctrl+C.

## Notes

- The program uses JACK's real-time audio processing capabilities
- FFT processing is done using the FFTW3 library
- Audio processing is done in blocks of the size determined by JACK's buffer size 