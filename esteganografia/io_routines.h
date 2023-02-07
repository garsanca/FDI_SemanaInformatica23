#ifndef IO_ROUTINES
#define IO_ROUTINES

typedef unsigned char uint8_t;

uint8_t* loadPNG(char *file, int *w, int *h);
void savePNG(char *file, uint8_t *data, int w, int h);

void get_msg(char *file_name, char **msg, int *msg_len);
void msg2logo(char *file_name, char *msg_decoded, int msg_len);

#endif
