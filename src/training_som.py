from main import *

import re
import os

def convert_X_to_dict(X):
    ret = {}
    ret['xdim'] = len(X)
    ret['ydim'] = 1
    ret['vec_dim'] = len(X[0])
    ret['arr'] = np.array(X)
    return ret

datasets = ['chainlink', '10clusters']
images = ['stick_figure.png','round.jpg','walking_icon.png']
size = ['small','large']
n_epochs = 10
#n_epochs = 500

parameters = {'chainlink': [1,5,1,10], '10clusters': [2,5,2,10]}
som_size = {'small': [40,20], 'large': [100,60]}


for dataset in datasets:
    for image in images:
        for s in size:
            image_name = re.split('\.', image)[0]
            print(dataset, ' - ', image_name, ' - ', s)
            if n_epochs == 10:
                idata_save_path = f"../trained_som/Epochs-10/idata_{dataset}_{image_name}_{s}.npy"
                weights_save_path = f"../trained_som/Epochs-10/weights_{dataset}_{image_name}_{s}.npy"
            elif n_epochs == 500:
                idata_save_path = f"../trained_som/Epochs-500/idata_{dataset}_{image_name}_{s}.npy"
                weights_save_path = f"../trained_som/Epochs-500/weights_{dataset}_{image_name}_{s}.npy"
            else:
                idata_save_path = f"../trained_som/idata_{dataset}_{image_name}_{s}.npy"
                weights_save_path = f"../trained_som/weights_{dataset}_{image_name}_{s}.npy"
            if os.path.exists(idata_save_path) and os.path.exists(weights_save_path):
                print('SOM already trained')
            else:
                som = som_test(image, dataset, som_size[s][0], som_size[s][1],
                           parameters[dataset][0], parameters[dataset][1],
                           parameters[dataset][2], parameters[dataset][3],
                           n_epochs = n_epochs)
                X = som.X
                idata = convert_X_to_dict(X)
                weights = som.export_weightlist()
                np.save(idata_save_path, idata)
                np.save(weights_save_path, weights)
                
