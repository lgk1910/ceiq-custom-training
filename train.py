import pickle
from sklearn.svm import SVR
import argparse
import os

parser = argparse.ArgumentParser(description='Image preprocessing for CEIQ training')
parser.add_argument('-o', '--output_model_path', default='CEIQ_model.pickle', type=str,
                    help='path of the output model')

args = parser.parse_args()

with open('Xs.pickle', 'rb') as f:
    Xs = pickle.load(f)
with open('ys.pickle', 'rb') as f:
    ys = pickle.load(f)
    
CEIQ_model = SVR(kernel='linear')
CEIQ_model.fit(Xs, ys)

print('Saving trained model...')
with open(args.output_model_path, 'wb') as f:
    pickle.dump(CEIQ_model, f)
    
print('Saved successfully')