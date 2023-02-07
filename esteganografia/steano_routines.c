
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>

#include "io_routines.h"
#include "steano_routines.h"

# define M_PI           3.14159265358979323846  /* pi */


void im2imRGB(uint8_t *im, int w, int h, t_sRGB *imRGB)
{
	imRGB->w = w;
	imRGB->h = h;

	for (int i=0; i<h; i++)
		for (int j=0; j<w; j++)
		{
			imRGB->R[i*w+j] = im[3*(i*w+j)  ];
			imRGB->G[i*w+j] = im[3*(i*w+j)+1];  
			imRGB->B[i*w+j] = im[3*(i*w+j)+2];    
		}                    
}

void imRGB2im(t_sRGB *imRGB, uint8_t *im, int *w, int *h)
{
	int w_ = imRGB->w;
	*w = imRGB->w;
	*h = imRGB->h;

	for (int i=0; i<*h; i++)
		for (int j=0; j<*w; j++)
		{
			im[3*(i*w_+j)  ] = imRGB->R[i*w_+j];
			im[3*(i*w_+j)+1] = imRGB->G[i*w_+j];  
			im[3*(i*w_+j)+2] = imRGB->B[i*w_+j];    
		}                    
}

//funtion for translate from RGB to YCbCr
void rgb2ycbcr(t_sRGB *in, t_sYCrCb *out)
{

	int w = in->w;
	out->w = in->w;
	out->h = in->h;

	for (int i = 0; i < in->h; i++) {
		for (int j = 0; j < in->w; j++) {

			// Use standard coeficient
			out->Y[i*w+j]  =         0.299*in->R[i*w+j]     + 0.587*in->G[i*w+j]      + 0.114*in->B[i*w+j];
			out->Cr[i*w+j] = 128.0 - 0.168736*in->R[i*w+j]  - 0.3331264*in->G[i*w+j]  + 0.5*in->B[i*w+j] ;
			out->Cb[i*w+j] = 128.0 + 0.5*in->R[i*w+j]       - 0.418688*in->G[i*w+j]   - 0.081312*in->B[i*w+j];
		}
	}
}

//function for translate YCbCr to RGB
void ycbcr2rgb(t_sYCrCb *in, t_sRGB *out){

	int w = in->w;
	out->w = in->w;
	out->h = in->h;

	for (int i = 0; i < in->h; i++) {
		for (int j = 0; j < in->w; j++) {

			// Use standard coeficient
			out->R[i*w+j] = in->Y[i*w+j]                                 + 1.402*(in->Cb[i*w+j]-128.0);
			out->G[i*w+j] = in->Y[i*w+j] - 0.34414*(in->Cr[i*w+j]-128.0) - 0.71414*(in->Cb[i*w+j]-128.0); 
			out->B[i*w+j] = in->Y[i*w+j] + 1.772*(in->Cr[i*w+j]-128.0);
			
			// After translate we must check if RGB component is in [0...255]
			if (out->R[i*w+j] < 0) out->R[i*w+j] = 0;
			else if (out->R[i*w+j] > 255) out->R[i*w+j] = 255;

			if (out->G[i*w+j] < 0) out->G[i*w+j] = 0;
			else if (out->G[i*w+j] > 255) out->G[i*w+j] = 255;

			if (out->B[i*w+j] < 0) out->B[i*w+j]= 0;
			else if (out->B[i*w+j] > 255) out->B[i*w+j] = 255;
		}
	}
}

void get_dct8x8_params(float *mcosine, float *alpha)
{
	int bM = 8;
	int bN = 8;

	for (int i = 0; i < bM; i++)
		for (int j = 0; j < bN; j++)
			mcosine[i*bN+j] = cos(((2*i+1)*M_PI*j)/(2*bM));

	alpha[0] = 1 / sqrt(bM * 1.0f);
	for (int i = 1; i < bM; i++)
		alpha[i] = sqrt(2.0f) / sqrt(bM * 1.0f);
}


//function for DCT. Picture divide block size 8x8
void dct8x8_2d(float *in, float *out, int width, int height, float *mcosine, float *alpha)
{
	int bM=8;
	int bN=8;

	for(int bi=0; bi<height/bM; bi++)
	{
		int stride_i = bi * bM;
		for(int bj=0; bj<width/bN; bj++)
		{
			int stride_j = bj * bN;
			for (int i=0; i<bM; i++)
			{
				for (int j=0; j<bN; j++)
				{
					float tmp = 0.0;
					for (int ii=0; ii < bM; ii++) 
					{
						for (int jj=0; jj < bN; jj++)
							tmp += in[(stride_i+ii)*width + stride_j+jj] * mcosine[ii*bN+i]*mcosine[jj*bN+j];
					}
					out[(stride_i+i)*width + stride_j+j] = tmp*alpha[i]*alpha[j];
				}
			}
		}
	}
}

void idct8x8_2d(float *in, float *out, int width, int height, float *mcosine, float *alpha)
{
	int bM=8;
	int bN=8;

	for(int bi=0; bi<height/bM; bi++)
	{
		int stride_i = bi * bM;
		for(int bj=0; bj<width/bN; bj++)
		{
			int stride_j = bj * bN;
			for (int i=0; i<bM; i++)
			{
				for (int j=0; j<bN; j++)
				{
					float tmp = 0.0;
					for (int ii=0; ii < bM; ii++) 
					{
						for (int jj=0; jj < bN; jj++)
							tmp += in[(stride_i+ii)*width + stride_j+jj] * mcosine[i*bN+ii]*mcosine[j*bN+jj]*alpha[ii]*alpha[jj];
					}
					out[(stride_i+i)*width + stride_j+j] = tmp;
				}
			}
		}
	}
}

void insert_msg(float *img, int width, int height, char *msg, int msg_length)
{
	int i_insert=3;
	int j_insert=4;

	int bM=8;
	int bN=8;
		
	int bsI = height/bM;
	int bsJ = width/bN;
	int bi = 0;
	int bj = 0;
	
	if(bsI*bsJ<msg_length*8)
		printf("Image not enough to save message!!!\n");

	for(int c=0; c<msg_length; c++)
		for(int b=0; b<8; b++)
		{
			char ch = msg[c];
			char bit = (ch&(1<<b))>>b;
			
			int stride_i = bi * bM;
			int stride_j = bj * bN;
			float tmp = 0.0;
			for (int ii=0; ii < bM; ii++) 
			{
				for (int jj=0; jj < bN; jj++)
					tmp += img[(stride_i+ii)*width + stride_j+jj];
			}
			float mean = tmp/(bM*bN);
			
//			img[(bi+i_insert)*width + bj+j_insert] = (float)(bit)*img[(bi+i_insert)*width + bj+j_insert];

			if (bit) 
				img[(stride_i+i_insert)*width + stride_j+j_insert] = fabsf(mean); //+
			else
				img[(stride_i+i_insert)*width + stride_j+j_insert] = -1.0f*fabsf(mean); //-


			bj++;
			if (bj>=bsJ){
				bj=0;
				bi++;
			}
		}
}

void extract_msg(float *img, int width, int height, char *msg, int msg_length)
{
	int i_insert=3;
	int j_insert=4;

	int bM=8;
	int bN=8;
		
	int bsI = height/bM;
	int bsJ = width/bN;
	int bi = 0;
	int bj = 0;
	
	for(int c=0; c<msg_length; c++){
		char ch=0;

		for(int b=0; b<8; b++)
		{
			int bit; 

			int stride_i = bi * bM;
			int stride_j = bj * bN;
			float tmp = 0.0;
			for (int ii=0; ii < bM; ii++) 
			{
				for (int jj=0; jj < bN; jj++)
					tmp += img[(stride_i+ii)*width + stride_j+jj];
			}
			float mean = tmp/(bM*bN);


			if (img[(stride_i+i_insert)*width + stride_j+j_insert]>0.5f*mean)
				bit = 1;
			else
				bit = 0;

			ch = (bit<<b)|ch;
	
			bj++;
			if (bj>=bsJ){
				bj=0;
				bi++;
			}
		}
		msg[c] = ch;
	}
}

void encoder(char *file_in, char *file_out, char *msg, int msg_len)
{

	int w, h, bytesPerPixel;

	// im corresponds to R0G0B0,R1G1B1.....
	uint8_t *im = (uint8_t*)loadPNG(file_in, &w, &h);
    
	// Create imRGB & imYCrCb
	uint8_t *im_out = malloc(3*w*h*sizeof(uint8_t));
	t_sRGB imRGB;
	imRGB.R = malloc(w*h*sizeof(float));
	imRGB.G = malloc(w*h*sizeof(float));
	imRGB.B = malloc(w*h*sizeof(float));
	t_sYCrCb imYCrCb;
	imYCrCb.Y  = malloc(w*h*sizeof(float));
	imYCrCb.Cr = malloc(w*h*sizeof(float));
	imYCrCb.Cb = malloc(w*h*sizeof(float));
	float *Ydct= malloc(w*h*sizeof(float));

	float *mcosine = malloc(8*8*sizeof(float));
	float *alpha = malloc(8*sizeof(float));
	get_dct8x8_params(mcosine, alpha);

	double start = omp_get_wtime();

	im2imRGB(im, w, h, &imRGB);
	rgb2ycbcr(&imRGB, &imYCrCb);
	dct8x8_2d(imYCrCb.Y, Ydct, imYCrCb.w, imYCrCb.h, mcosine, alpha);

	// Insert Message		
	insert_msg(Ydct, imYCrCb.w, imYCrCb.h, msg, msg_len);

	idct8x8_2d(Ydct, imYCrCb.Y, imYCrCb.w, imYCrCb.h, mcosine, alpha);
   	ycbcr2rgb(&imYCrCb, &imRGB);
	imRGB2im(&imRGB, im_out, &w, &h);

	double stop = omp_get_wtime();
	printf("Encoding time=%f sec.\n", stop-start);

	savePNG(file_out, im_out, w, h);

	free(imRGB.R); free(imRGB.G); free(imRGB.B);
	free(imYCrCb.Y); free(imYCrCb.Cr); free(imYCrCb.Cb);
	free(Ydct);
	free(mcosine); free(alpha);
}

void decoder(char *file_in, char *msg_decoded, int msg_len)
{

	int w, h, bytesPerPixel;

	// im corresponds to R0G0B0,R1G1B1.....
	uint8_t *im = (uint8_t*)loadPNG(file_in, &w, &h);
    
	// Create imRGB & imYCrCb
	uint8_t *im_out = malloc(3*w*h*sizeof(uint8_t));
	t_sRGB imRGB;
	imRGB.R = malloc(w*h*sizeof(float));
	imRGB.G = malloc(w*h*sizeof(float));
	imRGB.B = malloc(w*h*sizeof(float));
	t_sYCrCb imYCrCb;
	imYCrCb.Y  = malloc(w*h*sizeof(float));
	imYCrCb.Cr = malloc(w*h*sizeof(float));
	imYCrCb.Cb = malloc(w*h*sizeof(float));
	float *Ydct= malloc(w*h*sizeof(float));

	float *mcosine = malloc(8*8*sizeof(float));
	float *alpha = malloc(8*sizeof(float));
	get_dct8x8_params(mcosine, alpha);

	double start = omp_get_wtime();

	im2imRGB(im, w, h, &imRGB);
	rgb2ycbcr(&imRGB, &imYCrCb);
	dct8x8_2d(imYCrCb.Y, Ydct, imYCrCb.w, imYCrCb.h, mcosine, alpha);
		
	extract_msg(Ydct, imYCrCb.w, imYCrCb.h, msg_decoded, msg_len);

	double stop = omp_get_wtime();
	printf("Decoding time=%f sec.\n", stop-start);

	free(imRGB.R); free(imRGB.G); free(imRGB.B);
	free(imYCrCb.Y); free(imYCrCb.Cr); free(imYCrCb.Cb);
	free(Ydct);
	free(mcosine); free(alpha);
}
