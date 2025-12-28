#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>

#include <jack/jack.h>
#include <jack/ringbuffer.h>
#include <sndfile.h> // Library for writing WAV files

jack_port_t *input_port;
jack_port_t *output_port;
jack_client_t *client;
jack_ringbuffer_t *ringbuffer;

// Flag to stop the main loop
volatile int keep_running = 1;

void signal_handler(int sig) {
    keep_running = 0;
}

/**
 * The Real-Time Callback
 * We ONLY write to memory here (Ringbuffer). No disk writing!
 */
int jack_callback (jack_nframes_t nframes, void *arg){
    jack_default_audio_sample_t *in, *out;
    
    in = jack_port_get_buffer (input_port, nframes);
    out = jack_port_get_buffer (output_port, nframes);
    
    // 1. Copy input to output (Hear the audio)
    memcpy (out, in, nframes * sizeof (jack_default_audio_sample_t));

    // 2. Push input to Ringbuffer (Save the audio)
    // We check if there is enough space first to avoid overwriting
    size_t bytes_to_write = nframes * sizeof(jack_default_audio_sample_t);
    
    if (jack_ringbuffer_write_space(ringbuffer) >= bytes_to_write) {
        jack_ringbuffer_write(ringbuffer, (char *)in, bytes_to_write);
    }
    
    return 0;
}

int main (int argc, char *argv[]) {
    // ... [Standard JACK setup code same as before] ...
    const char *client_name = "recorder_client";
    jack_options_t options = JackNoStartServer;
    jack_status_t status;
    client = jack_client_open (client_name, options, &status);
    if (client == NULL) exit(1);

    // --- SETUP RINGBUFFER ---
    // Allocate enough memory for ~5 seconds of audio buffering to be safe
    // 48000 samples * 4 bytes (float) * 5 seconds = ~960kb
    size_t ringbuffer_size = 48000 * sizeof(float) * 5; 
    ringbuffer = jack_ringbuffer_create(ringbuffer_size);

    // --- SETUP WAV FILE (libsndfile) ---
    SF_INFO sfinfo;
    sfinfo.channels = 1;             // Mono
    sfinfo.samplerate = 48000;       // Standard JACK rate (should technically ask jack_get_sample_rate)
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT; // 32-bit Float WAV
    
    SNDFILE *outfile = sf_open("recording.wav", SFM_WRITE, &sfinfo);
    if (!outfile) {
        printf("Error opening output file!\n");
        exit(1);
    }

    // Register ports and callback
    jack_set_process_callback (client, jack_callback, 0);
    input_port = jack_port_register (client, "input", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    output_port = jack_port_register (client, "output", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    jack_activate (client);

    // [Auto-connect ports logic goes here...]

    // --- MAIN LOOP (DISK WRITER) ---
    signal(SIGINT, signal_handler); // Catch Ctrl+C
    printf("Recording to recording.wav... Press Ctrl+C to stop.\n");

    // Temporary buffer for reading from ringbuffer
    // We process in chunks of 1024 samples
    size_t chunk_size = 1024;
    float *read_buffer = malloc(chunk_size * sizeof(float));

    while (keep_running) {
        // Check how many bytes are waiting in the ringbuffer
        size_t bytes_available = jack_ringbuffer_read_space(ringbuffer);
        
        // If we have at least one chunk of data, write it to disk
        if (bytes_available >= chunk_size * sizeof(float)) {
            
            // 1. Pull from Ringbuffer
            jack_ringbuffer_read(ringbuffer, (char*)read_buffer, chunk_size * sizeof(float));
            
            // 2. Write to Disk
            sf_write_float(outfile, read_buffer, chunk_size);
        } else {
            // If no data, sleep a tiny bit to save CPU
            usleep(1000); // 1ms
        }
    }

    // --- CLEANUP ---
    printf("Saving and closing...\n");
    sf_close(outfile);           // Close WAV file
    free(read_buffer);
    jack_ringbuffer_free(ringbuffer);
    jack_client_close(client);
    
    return 0;
}
