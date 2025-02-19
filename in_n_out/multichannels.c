/**
 * A simple 1-input to 1-output JACK client.
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <jack/jack.h>

jack_port_t ** input_ports;
jack_port_t ** output_ports;
jack_client_t *client;

int channels;

/**
 * The process callback for this JACK application is called in a
 * special realtime thread once for each audio cycle.
 *
 * This client does nothing more than copy data from its input
 * port to its output port. It will exit when stopped by 
 * the user (e.g. using Ctrl-C on a unix-ish operating system)
 */
int jack_callback (jack_nframes_t nframes, void *arg){
  jack_default_audio_sample_t *in[channels], *out[channels];
  int i,j;
  
  for ( i = 0; i < channels; ++i)
  {
    in[i] = jack_port_get_buffer(input_ports[i], nframes);
    out[i] = jack_port_get_buffer(output_ports[i], nframes);
  }

  for ( i = 0; i < channels; ++i)
  {
    for ( j = 0; j < nframes; ++j)
    {
      out[i][j] = in[i][j] ;
    }
  }
  
  

  return 0;
}


/**
 * JACK calls this shutdown_callback if the server ever shuts down or
 * decides to disconnect the client.
 */
void jack_shutdown (void *arg){
  exit (1);
}


int main (int argc, char *argv[]) {
  if (argc < 2){
    printf("Number channels is required.\n");
    exit(1);
  }

  channels = atoi(argv[1])

  const char *client_name = "in_to_out";
  jack_options_t options = JackNoStartServer;
  jack_status_t status;
  
  /* open a client connection to the JACK server */
  client = jack_client_open (client_name, options, &status);
  if (client == NULL){
    /* if connection failed, say why */
    printf ("jack_client_open() failed, status = 0x%2.0x\n", status);
    if (status & JackServerFailed) {
      printf ("Unable to connect to JACK server.\n");
    }
    exit (1);
  }
  
  /* if connection was successful, check if the name we proposed is not in use */
  if (status & JackNameNotUnique){
    client_name = jack_get_client_name(client);
    printf ("Warning: other agent with our name is running, `%s' has been assigned to us.\n", client_name);
  }
  
  /* tell the JACK server to call 'jack_callback()' whenever there is work to be done. */
  jack_set_process_callback (client, jack_callback, 0);
  
  
  /* tell the JACK server to call 'jack_shutdown()' if it ever shuts down,
     either entirely, or if it just decides to stop calling us. */
  jack_on_shutdown (client, jack_shutdown, 0);
  
  
  /* display the current sample rate. */
  printf ("Engine sample rate: %d\n", jack_get_sample_rate (client));
  
  
  /* display the current window size. */
  printf ("Engine window size: %d\n", jack_get_buffer_size (client));
  
  
  /* create the agent input port */
  int i;
  char port_name[20];

  input_ports = malloc(channels * sizeof(jack_port_t *))
  for (i = 0; i < channels; ++i){     
    sprintf(port_name,"input%d",i+1)
    input_ports[i] = jack_port_register (client, port_name, JACK_DEFAULT_AUDIO_TYPE,JackPortIsInput, 0); 
    /* check that both ports were created succesfully */
    if (input_ports[i] == NULL) {
      printf("Could not create agent ports. Have we reached the maximum amount of JACK agent ports?\n");
      exit (1);
    }
  }
  
  
  /* create the agent output port */
  output_ports = malloc(channels * sizeof(jack_port_t *))
  for (i = 0; i < channels; ++i){     
    sprintf(port_name,"output%d",i+1)
    output_ports[i] = jack_port_register (client, port_name, JACK_DEFAULT_AUDIO_TYPE,JackPortIsOutput, 0); 
    /* check that both ports were created succesfully */
    if (output_ports[i] == NULL) {
      printf("Could not create agent ports. Have we reached the maximum amount of JACK agent ports?\n");
      exit (1);
    }
  }
  
  
  /* Tell the JACK server that we are ready to roll.
     Our jack_callback() callback will start running now. */
  if (jack_activate (client)) {
    printf ("Cannot activate client.");
    exit (1);
  }
  
  printf ("Agent activated.\n");
   
  // free serverports_names variable, we're not going to use it again
  free (serverports_names);
  
  
  printf ("done.\n");
  /* keep running until stopped by the user */
  sleep (-1);
  
  
  /* this is never reached but if the program
     had some other way to exit besides being killed,
     they would be important to call.
  */
  jack_client_close (client);
  exit (0);
}