from CEIQ import CEIQ
import cv2
import argparse

parser = argparse.ArgumentParser(description='Image preprocessing for CEIQ training')
parser.add_argument('-m', '--model_path', default='CEIQ_model.pickle', type=str,
                    help='name of the output model')

args = parser.parse_args()

model = CEIQ(args.model_path)
results0 = model.predict(['test_imgs/1.png', 'test_imgs/2.png'], 0) # 'option' is set to 0 to indicate prediction from paths

img1 = cv2.imread('test_imgs/1.png')
img2 = cv2.imread('test_imgs/2.png')
results1 = model.predict([img1, img2], 1) # 'option' is set to 1 to indicate prediction from BGR matrix representations of images

print(results0, results1) # the two outputs are supposed to be the same