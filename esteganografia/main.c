#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "steano_routines.h"
#include "io_routines.h"

int main(int argc, char **argv)
{
	char file_in[20], file_out[20], file_logo[20];

	if (argc<3){
		printf("./exec image_in.png logo.png image_out.png\n");
		return(-1);
	}

	strcpy(file_in, argv[1]);
	strcpy(file_logo, argv[2]);
	strcpy(file_out, argv[3]);
	

	char *msg, *msg_decoded;
	int msg_len;
	get_msg(file_logo, &msg, &msg_len);

	// Encode the msg into image
	encoder(file_in, file_out, msg, msg_len);

	// Extract msg from image
	msg_decoded = malloc(msg_len);
	decoder(file_out, msg_decoded, msg_len);
	msg2logo("logo_out.png", msg_decoded, msg_len);

	return(0);
}
