

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "io_routines.h"

void get_msg(char *file_name, char **msg, int *msg_len)
{
	int w, h, bytesPerPixel;

	uint8_t *im = (uint8_t*)stbi_load(file_name, &w, &h, &bytesPerPixel, STBI_rgb);
	*msg_len = w*h/8;

	char *msg_ = malloc((*msg_len)*sizeof(char));

	int ib = 0;
	int imsg = 0;
	char byte=0;

	for (int i=0; i<h; i++){
		for (int j=0; j<w; j++) {
			char bit;

			if (im[3*(i*w+j)]){
				bit = 1;
				byte = (bit<<ib)|byte;
			} else bit = 0;
			
			ib++;
			if (ib>=8)
			{
				msg_[imsg++] = byte;
				ib = 0;
				byte = 0;
			}
		}
	}
	*msg = msg_;
}

void msg2logo(char *file_name, char *msg_decoded, int msg_len)
{
	uint8_t *im  = malloc(3*8*msg_len*sizeof(uint8_t));

	int logo_size = (int)roundf(sqrtf(8.0f*msg_len));

	int imsg = 0;
	uint8_t pixel;
	for (int imsg=0; imsg<msg_len; imsg++){
		char byte = msg_decoded[imsg];
		for (int ib=0; ib<8; ib++){
			char bit = (0x01)&(byte>>ib);
			if (bit) pixel = 255;
			else pixel = 0;

			int i = (8*imsg+ib)/logo_size;
			int j = (8*imsg+ib)%logo_size;

			im[3*(i*logo_size+j)  ] = pixel;
			im[3*(i*logo_size+j)+1] = pixel;
			im[3*(i*logo_size+j)+2] = pixel;
		}
	}

	savePNG(file_name, im, logo_size, logo_size);
}

uint8_t* loadPNG(char *file_in, int *w, int *h)
{
	int bytesPerPixel;

	// im corresponds to R0G0B0,R1G1B1.....
	uint8_t *im = (uint8_t*)stbi_load(file_in, w, h, &bytesPerPixel, STBI_rgb);

	return im;
} 

void savePNG(char *file, uint8_t *data, int w, int h)
{
	stbi_write_png(file, w, h, 3, data, 3*w);
}
