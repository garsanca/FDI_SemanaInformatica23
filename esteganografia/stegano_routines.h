#ifndef STEANO_ROUTINES
#define STEANO_ROUTINES

struct sRGB {
    float *R;
    float *G;
    float *B;
    int w;
    int h;
};
typedef struct sRGB t_sRGB;

struct sYCrCb {
    float *Y;
    float *Cr;
    float *Cb;
    int w;
    int h;
};
typedef struct sYCrCb t_sYCrCb;

void encoder(char *file_in, char *file_out, char *msg, int msg_len);
void decoder(char *file_in, char *msg_decoded, int msg_len);

#endif

