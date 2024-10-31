from bench3dgs.compression.codec import Codec

import imagecodecs

class JpegXlCodec(Codec):

    def encode_image(self, image, out_file, **kwargs):
        imagecodecs.imwrite(out_file, image, **kwargs)

    def decode_image(self, file_name):
        return imagecodecs.imread(file_name)

    def file_ending(self):
        return "jxl"