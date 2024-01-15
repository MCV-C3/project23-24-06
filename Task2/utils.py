#from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_image_patches_db(in_directory,out_directory,patch_size=64):
  if not os.path.exists(out_directory):
      os.makedirs(out_directory)
 
  total = 2688
  count = 0  
  for split_dir in os.listdir(in_directory):
    if not os.path.exists(os.path.join(out_directory,split_dir)):
      os.makedirs(os.path.join(out_directory,split_dir))
  
    for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
      if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
        os.makedirs(os.path.join(out_directory,split_dir,class_dir))
  
      for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
        count += 1
        im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
        print(im.size)
        print('Processed images: '+str(count)+' / '+str(total), end='\r')
        patches = image.extract_patches_2d(np.array(im), (64, 64), max_patches=1)
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')

def confusion(actual, predicted, classes):
  conf = np.zeros((len(classes), len(classes)), dtype=np.float64)

  mp = {}
  for i, cls in enumerate(classes):
      mp[cls] = i

  for a, p in zip(actual, predicted):
      conf[mp[a], mp[p]] += 1

  return conf

#CONFUSION MATRIX CODE
def confusion_matrix(classes, predictions, gt_labels, path):
  conf = confusion(gt_labels, predictions, classes)
  ax = sns.heatmap(conf, cmap='YlGnBu', xticklabels=classes, yticklabels=classes, square=True, annot=True)
  ax.set_xlabel('Predicted Class')
  ax.set_ylabel('Actual Class')
  ax.set_title('Confusion Matrix')
  plt.savefig(path)
