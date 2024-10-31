from bench3dgs.compression.codec import Codec

import numpy as np
import cv2

# dtype: uint8, uint16

class PNGCodec(Codec):

    def encode_image(self, image, out_file, dtype):

        match dtype:
            case "uint8":
                image = image * 255
                image = image.astype("uint8")
            case "uint16":
                image = image * 65535
                image = image.astype("uint16")

        cv2.imwrite(out_file, image)

    def decode_image(self, file_name):
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        match img.dtype:
            case np.uint8:
                img = img / 255
            case np.uint16:
                img = img / 65535
        return img

    def file_ending(self):
        return "png"