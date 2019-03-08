import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
# Main slim library
from lib_1.contour import *
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_dilation
from tensorflow.contrib import slim
from nets import inception_v4

from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from skimage.transform import resize
from imgaug import augmenters as iaa
from skimage import exposure
from datasets import dataset_utils
from sklearn.utils import shuffle

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

np.random.seed(13)

url = "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
SNAPSHOT_FILE = "/app/trained_checkpoints/inception_v4/lung_output12/snapshot_23.ckpt"
LAST_SNAPSHOT_FILE = "/app/trained_checkpoints/inception_v4/lung_output12/snapshot_22.ckpt"

SNAPSHOT_DIR_CH = "/app/trained_checkpoints/inception_v4/lung_output12/"
SNAPSHOT_DIR_OU =  '/app/output/inception_v4/lung_output12/'
PRETRAINED_SNAPSHOT_FILE = '/app/pretrained_checkpoints/inception_v4.ckpt'
PRETRAINED_SNAPSHOT_DIR = '/app/pretrained_checkpoints/'

if not tf.gfile.Exists(PRETRAINED_SNAPSHOT_FILE):
    tf.gfile.MakeDirs(PRETRAINED_SNAPSHOT_DIR)
    dataset_utils.download_and_uncompress_tarball(url, PRETRAINED_SNAPSHOT_DIR)

if not tf.gfile.Exists(SNAPSHOT_DIR_CH):
    tf.gfile.MakeDirs(SNAPSHOT_DIR_CH)

if not tf.gfile.Exists(SNAPSHOT_DIR_OU):
    tf.gfile.MakeDirs(SNAPSHOT_DIR_OU)


#tf.reset_default_graph()
# CT image path
ct_image_path = '/app/SRBT/PRE_SBRT_CTpet/CTpet_AO_max_area_image/'
ct_contour_path = '/app/SRBT/PRE_SBRT_CTpet/CTpet_AO_max_area_tumor_contour/'

# High-res CT image path
highres_ct_image_path = '/app/SRBT/PRE_SBRT_CT/CTpet_AO_max_area_image/'
highres_ct_contour_path = '/app/SRBT/PRE_SBRT_CT/CTpet_AO_max_area_tumor_contour/'

# POST CT image path
post_ct_image_path = '/app/SRBT/POST_SBRT_CT/CTpet_AO_max_area_image/'
post_ct_contour_path = '/app/SRBT/POST_SBRT_CT/CTpet_AO_max_area_tumor_contour/'

# PET image path
pet_image_path = '/app/SRBT/PRE_SBRT_PETct/PETct_AO_max_area_image/'
pet_contour_path = '/app/SRBT/PRE_SBRT_PETct/PETct_AO_max_area_tumor_contour/'


output_file = '/app/output/inception_v4/lung_output12/end2end_out_mod_23.txt'
output_file1 = '/app/output/inception_v4/lung_output12/fold_sample_ind_23.txt'
log_output = '/app/output/inception_v4/lung_output12/output_23'


print_tensors_in_checkpoint_file(file_name=PRETRAINED_SNAPSHOT_FILE, tensor_name='', all_tensors=False, all_tensor_names=True)

filelist = [f.split('.')[0].split('_')[1] for f in os.listdir(ct_image_path) if f.endswith(".npy")]
#print(filelist)
csv_info = pd.read_csv('/app/SRBT/SBRT_mod.csv')

csv_info = csv_info[csv_info['Patient_ID'].isin(filelist)]
csv_info = csv_info[csv_info['Count_of_the_patients'] == 1]

csv_info['Start_Date'] = pd.to_datetime(csv_info.Start_Date, format= '%d-%m-%Y')
csv_info['End_Death'] = pd.to_datetime(csv_info.End_Death, format= '%d-%m-%Y')
csv_info['End_REC'] = pd.to_datetime(csv_info.End_REC, format= '%d-%m-%Y')
csv_info['End_DIST'] = pd.to_datetime(csv_info.End_DIST, format= '%d-%m-%Y')
csv_info['End_LOBE'] = pd.to_datetime(csv_info.End_LOBE, format= '%d-%m-%Y')
csv_info['End_LOC'] = pd.to_datetime(csv_info.End_LOC, format= '%d-%m-%Y')
csv_info['End_REG'] = pd.to_datetime(csv_info.End_REG, format= '%d-%m-%Y')


csv_info['Death_Days_diff'] = (csv_info['End_Death'] - csv_info['Start_Date']).dt.days
csv_info['Death_months_diff'] = ((csv_info['End_Death'] - csv_info['Start_Date'])/np.timedelta64(1, 'M'))
csv_info['Death_months_diff'] = csv_info['Death_months_diff'].astype(int)

csv_info['REC_Days_diff'] = (csv_info['End_REC'] - csv_info['Start_Date']).dt.days
csv_info['REC_months_diff'] = ((csv_info['End_REC'] - csv_info['Start_Date'])/np.timedelta64(1, 'M'))
csv_info['REC_months_diff'] = csv_info['REC_months_diff'].astype(int)

csv_info['DIST_Days_diff'] = (csv_info['End_DIST'] - csv_info['Start_Date']).dt.days
csv_info['DIST_months_diff'] = ((csv_info['End_DIST'] - csv_info['Start_Date'])/np.timedelta64(1, 'M'))
csv_info['DIST_months_diff'] = csv_info['DIST_months_diff'].astype(int)

csv_info['LOBE_Days_diff'] = (csv_info['End_LOBE'] - csv_info['Start_Date']).dt.days
csv_info['LOBE_months_diff'] = ((csv_info['End_LOBE'] - csv_info['Start_Date'])/np.timedelta64(1, 'M'))
csv_info['LOBE_months_diff'] = csv_info['LOBE_months_diff'].astype(int)

csv_info['LOC_Days_diff'] = (csv_info['End_LOC'] - csv_info['Start_Date']).dt.days
csv_info['LOC_months_diff'] = ((csv_info['End_LOC'] - csv_info['Start_Date'])/np.timedelta64(1, 'M'))
csv_info['LOC_months_diff'] = csv_info['LOC_months_diff'].astype(int)

csv_info['REG_Days_diff'] = (csv_info['End_REG'] - csv_info['Start_Date']).dt.days
csv_info['REG_months_diff'] = ((csv_info['End_REG'] - csv_info['Start_Date'])/np.timedelta64(1, 'M'))
csv_info['REG_months_diff'] = csv_info['REG_months_diff'].astype(int)

csv_info.ix[csv_info['HIST'].isnull(), 'HIST'] = 0
csv_info.ix[csv_info['CAUSE'].isnull(), 'CAUSE'] = 0
csv_info['CAUSE'] = csv_info['CAUSE'].astype('int64')
print(csv_info['CAUSE'])

csv_info.ix[csv_info['SEX'] == 'M', 'SEX'] = 0
csv_info.ix[csv_info['SEX'] == 'F', 'SEX'] = 1


csv_info[['AGE','SEX','STAGE','HIST','BED']]=(csv_info[['AGE','SEX','STAGE','HIST','BED']]-csv_info[['AGE','SEX','STAGE','HIST','BED']].min())/(csv_info[['AGE','SEX','STAGE','HIST','BED']].max()-csv_info[['AGE','SEX','STAGE','HIST','BED']].min())


full_pet_image_dataset_path = list()
full_pet_contour_dataset_path = list()
full_ct_image_dataset_path = list()
full_ct_contour_dataset_path = list()
full_highres_ct_image_dataset_path = list()
full_highres_ct_contour_dataset_path = list()
full_post_ct_image_dataset_path = list()
full_post_ct_contour_dataset_path = list()

full_dataset_features = list()
full_dataset_labels = list()

for name, features, labels in zip(csv_info['Patient_ID'], np.array(csv_info[['AGE','SEX','STAGE','HIST','BED']])  ,np.array(csv_info[['Death_Days_diff','DEATH', 'REC_Days_diff', 'REC', 'DIST_Days_diff', 'DIST_REC','LOBE_Days_diff', 'LOBE_REC', 'LOC_Days_diff', 'LOC_REC', 'REG_Days_diff', 'REG_REC', 'CAUSE']])):
    #create pet dirt list
    full_pet_image_dataset_path.append(os.path.join(pet_image_path, 'ID_'+ str(name)+'.npy'))
    full_pet_contour_dataset_path.append(os.path.join(pet_contour_path, 'ID_'+ str(name)+'.npy'))
    #create std ct dirt list
    full_ct_image_dataset_path.append(os.path.join(ct_image_path, 'ID_'+ str(name)+'.npy'))
    full_ct_contour_dataset_path.append(os.path.join(ct_contour_path, 'ID_'+ str(name)+'.npy'))
    # create high res ct dirt list
    full_highres_ct_image_dataset_path.append(os.path.join(highres_ct_image_path, 'ID_'+ str(name)+'.npy'))
    full_highres_ct_contour_dataset_path.append(os.path.join(highres_ct_contour_path, 'ID_'+ str(name)+'.npy'))
    #create post ct dirt list
    full_post_ct_image_dataset_path.append(os.path.join(post_ct_image_path, 'ID_'+ str(name)+'.npy'))
    full_post_ct_contour_dataset_path.append(os.path.join(post_ct_contour_path, 'ID_'+ str(name)+'.npy'))
    #create feature and label dataset
    full_dataset_features.append([feature for feature in features])
    full_dataset_labels.append([label for label in labels])

full_pet_image_dataset_path = np.array(full_pet_image_dataset_path)
full_pet_contour_dataset_path = np.array(full_pet_contour_dataset_path)
full_ct_image_dataset_path = np.array(full_ct_image_dataset_path)
full_ct_contour_dataset_path = np.array(full_ct_contour_dataset_path)
full_highres_ct_image_dataset_path = np.array(full_highres_ct_image_dataset_path)
full_highres_ct_contour_dataset_path = np.array(full_highres_ct_contour_dataset_path)
full_post_ct_image_dataset_path = np.array(full_post_ct_image_dataset_path)
full_post_ct_contour_dataset_path = np.array(full_post_ct_contour_dataset_path)

full_dataset_features = np.array(full_dataset_features)
full_dataset_labels = np.array(full_dataset_labels)



full_pet_image_dataset_path,full_pet_contour_dataset_path,full_ct_image_dataset_path,full_ct_contour_dataset_path,full_highres_ct_image_dataset_path,full_highres_ct_contour_dataset_path,full_post_ct_image_dataset_path,full_post_ct_contour_dataset_path,full_dataset_features,full_dataset_labels = shuffle(full_pet_image_dataset_path,full_pet_contour_dataset_path,full_ct_image_dataset_path,full_ct_contour_dataset_path,full_highres_ct_image_dataset_path,full_highres_ct_contour_dataset_path,full_post_ct_image_dataset_path,full_post_ct_contour_dataset_path,full_dataset_features,full_dataset_labels, random_state=13)




def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def initialize_vars(session):
    # INITIALIZE VARS
    print('xxxxxxxxxxxxxxxxxxxxxx')
    if tf.train.checkpoint_exists(LAST_SNAPSHOT_FILE+ '_'+str(k_fold_num)) and False:
        print('xxxxxxxxxxxxxxxxxxxxxx')
        print("Loading from Last Checkpoint ", LAST_SNAPSHOT_FILE, k_fold_num)
        tf_saver.restore(session, LAST_SNAPSHOT_FILE + '_'+str(k_fold_num))


def CTpet1_initialize_vars(session):
    # INITIALIZE VARS

    if tf.train.checkpoint_exists(PRETRAINED_SNAPSHOT_FILE) and True:
        print("CTpet1 Initializing from imagenet Pretrained Weights", PRETRAINED_SNAPSHOT_FILE, k_fold_num)
        CTpet1_tf_imagenet_pretrained_saver.restore(session, PRETRAINED_SNAPSHOT_FILE)

    if tf.train.checkpoint_exists(LAST_SNAPSHOT_FILE+ '_'+str(k_fold_num)) and True:
        print("CTpet1 Loading from Last Checkpoint ", LAST_SNAPSHOT_FILE, k_fold_num)
        CTpet1_tf_saver.restore(session, LAST_SNAPSHOT_FILE + '_'+str(k_fold_num))

def CTpet2_initialize_vars(session):
    # INITIALIZE VARS

    if tf.train.checkpoint_exists(PRETRAINED_SNAPSHOT_FILE) and True:
        print("CTpet2 Initializing from imagenet Pretrained Weights", PRETRAINED_SNAPSHOT_FILE, k_fold_num)
        CTpet2_tf_imagenet_pretrained_saver.restore(session, PRETRAINED_SNAPSHOT_FILE)

    if tf.train.checkpoint_exists(LAST_SNAPSHOT_FILE+ '_'+str(k_fold_num)) and True:
        print("CTpet2 Loading from Main Checkpoint ", LAST_SNAPSHOT_FILE, k_fold_num)
        CTpet2_tf_saver.restore(session, LAST_SNAPSHOT_FILE + '_'+str(k_fold_num))

def CTpet3_initialize_vars(session):
    # INITIALIZE VARS

    if tf.train.checkpoint_exists(PRETRAINED_SNAPSHOT_FILE) and True:
        print("CTpet3 Initializing from imagenet Pretrained Weights", PRETRAINED_SNAPSHOT_FILE, k_fold_num)
        CTpet3_tf_imagenet_pretrained_saver.restore(session, PRETRAINED_SNAPSHOT_FILE)

    if tf.train.checkpoint_exists(LAST_SNAPSHOT_FILE+ '_'+str(k_fold_num)) and True:
        print("CTpet3 Loading from Main Checkpoint ", LAST_SNAPSHOT_FILE, k_fold_num)
        CTpet3_tf_saver.restore(session, LAST_SNAPSHOT_FILE + '_'+str(k_fold_num))

def CTpet4_initialize_vars(session):
    # INITIALIZE VARS

    if tf.train.checkpoint_exists(PRETRAINED_SNAPSHOT_FILE) and True:
        print("CTpet4 Initializing from imagenet Pretrained Weights", PRETRAINED_SNAPSHOT_FILE, k_fold_num)
        CTpet4_tf_imagenet_pretrained_saver.restore(session, PRETRAINED_SNAPSHOT_FILE)

    if tf.train.checkpoint_exists(LAST_SNAPSHOT_FILE+ '_'+str(k_fold_num)) and True:
        print("CTpet4 Loading from Main Checkpoint", LAST_SNAPSHOT_FILE, k_fold_num)
        CTpet1_tf_saver.restore(session, LAST_SNAPSHOT_FILE + '_'+str(k_fold_num))

def CTpet_fc_initialize_vars(session):
    # INITIALIZE VARS

    if tf.train.checkpoint_exists(LAST_SNAPSHOT_FILE+ '_'+str(k_fold_num)) and True:
        print(" CTpet_fc Loading from Main Checkpoint", LAST_SNAPSHOT_FILE, k_fold_num)
        CTpet_fc_tf_saver.restore(session, LAST_SNAPSHOT_FILE + '_'+str(k_fold_num))




def create_train( image_ct_dataset_path, contour_ct_dataset_path,image_highres_ct_dataset_path, contour_highres_ct_dataset_path,image_post_ct_dataset_path, contour_post_ct_dataset_path, dataset_features, dataset_labels):
    #print(image_pet_dataset_path)
    shape =(299,299,3)

    #print('image_ct_dataset_path >>>>>>>>>>>>>>>>>>>>>>',image_ct_dataset_path)
    # PRE-CT
    ct_image, ct_contour, slice_num1, slice_num2, slice_num3, slice_num4 = load_train_image( image_ct_dataset_path, contour_ct_dataset_path, shape)
    ct_batch_images1, ct_batch_images2, ct_batch_images3, ct_batch_images4 = format_train_image(ct_image,ct_contour, slice_num1, slice_num2, slice_num3, slice_num4, shape)

    #print('image_highres_ct_dataset_path >>>>>>>>>>>>>>>>>>>',image_highres_ct_dataset_path)
    # HIGHRES CT
    highres_ct_image, highres_ct_contour, slice_num1, slice_num2, slice_num3, slice_num4 = load_train_image( image_highres_ct_dataset_path, contour_highres_ct_dataset_path, shape)
    highres_ct_batch_images1, highres_ct_batch_images2, highres_ct_batch_images3, highres_ct_batch_images4 = format_train_image(highres_ct_image,highres_ct_contour, slice_num1, slice_num2, slice_num3, slice_num4, shape)

    #print('image_post_ct_dataset_path >>>>>>>>>>>>>>>>>',image_post_ct_dataset_path)
    # POST CT
    post_ct_image, post_ct_contour, slice_num1, slice_num2, slice_num3, slice_num4 = load_train_image( image_post_ct_dataset_path, contour_post_ct_dataset_path, shape)
    post_ct_batch_images1, post_ct_batch_images2, post_ct_batch_images3, post_ct_batch_images4 = format_train_image(post_ct_image, post_ct_contour, slice_num1, slice_num2, slice_num3, slice_num4, shape)




    ct_batch_images1 = np.float32(ct_batch_images1)
    ct_batch_images2 = np.float32(ct_batch_images2)
    ct_batch_images3 = np.float32(ct_batch_images3)
    ct_batch_images4 = np.float32(ct_batch_images4)

    highres_ct_batch_images1 = np.float32(highres_ct_batch_images1)
    highres_ct_batch_images2 = np.float32(highres_ct_batch_images2)
    highres_ct_batch_images3 = np.float32(highres_ct_batch_images3)
    highres_ct_batch_images4 = np.float32(highres_ct_batch_images4)

    post_ct_batch_images1 = np.float32(post_ct_batch_images1)
    post_ct_batch_images2 = np.float32(post_ct_batch_images2)
    post_ct_batch_images3 = np.float32(post_ct_batch_images3)
    post_ct_batch_images4 = np.float32(post_ct_batch_images4)


    dataset_features = np.float32(dataset_features)
    dataset_labels = np.int16(dataset_labels)

    return ct_batch_images1, ct_batch_images2, ct_batch_images3, ct_batch_images4, highres_ct_batch_images1, highres_ct_batch_images2, highres_ct_batch_images3, highres_ct_batch_images4,post_ct_batch_images1, post_ct_batch_images2, post_ct_batch_images3, post_ct_batch_images4, dataset_features , dataset_labels



def create_val( image_ct_dataset_path, contour_ct_dataset_path, dataset_features, dataset_labels):
    #print(image_pet_dataset_path)
    shape =(299,299,3)
    image, contour, slice_num1, slice_num2, slice_num3, slice_num4 = load_val_image(image_ct_dataset_path, contour_ct_dataset_path,shape)
    ct_batch_images1, ct_batch_images2, ct_batch_images3, ct_batch_images4 = format_val_image(image,contour, slice_num1, slice_num2, slice_num3, slice_num4, shape)

    ct_batch_images1 = np.float32(ct_batch_images1)
    ct_batch_images2 = np.float32(ct_batch_images2)
    ct_batch_images3 = np.float32(ct_batch_images3)
    ct_batch_images4 = np.float32(ct_batch_images4)

    dataset_features = np.float32(dataset_features)
    dataset_labels = np.int16(dataset_labels)


    return ct_batch_images1, ct_batch_images2, ct_batch_images3, ct_batch_images4, dataset_features , dataset_labels


def create_test( image_ct_dataset_path, contour_ct_dataset_path, dataset_features, dataset_labels):
    #print(image_pet_dataset_path)
    shape =(299,299,3)
    image, contour, slice_num1, slice_num2,slice_num3, slice_num4 = load_test_image(image_ct_dataset_path, contour_ct_dataset_path,shape)
    ct_batch_images1, ct_batch_images2, ct_batch_images3, ct_batch_images4 = format_test_image(image,contour, slice_num1, slice_num2, slice_num3, slice_num4, shape)

    ct_batch_images1 = np.float32(ct_batch_images1)
    ct_batch_images2 = np.float32(ct_batch_images2)
    ct_batch_images3 = np.float32(ct_batch_images3)
    ct_batch_images4 = np.float32(ct_batch_images4)

    dataset_features = np.float32(dataset_features)
    dataset_labels = np.int16(dataset_labels)


    return ct_batch_images1, ct_batch_images2, ct_batch_images3, ct_batch_images4, dataset_features , dataset_labels


def load_train_image(image_dataset_path, contour_dataset_path, shape):

    image = np.load(image_dataset_path.decode())
    contour = np.load(contour_dataset_path.decode())
    image = np.transpose(image, [1, 2, 0])
    contour = np.transpose(contour, [1, 2, 0])
    #print('image shape ', image.shape)
    #print('contour shape ', contour.shape)

    if contour.shape[2] == 3:
        #print('contour shape----------------------- ', contour.shape)
        image = np.concatenate([image, image], axis = 2)
        contour = np.concatenate([contour, contour], axis = 2)
        #print('New image shape 3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',image.shape, contour.shape)
    elif contour.shape[2] == 2:
        #print('contour shape----------------------- ', contour.shape)
        image = np.concatenate([image, image, image], axis = 2)
        contour = np.concatenate([contour, contour, contour], axis = 2)
        #print('New image shape 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',image.shape, contour.shape)

    elif contour.shape[2] == 1:
        #print('contour shape----------------------- ', contour.shape)
        image = np.concatenate([image, image, image,image,image, image], axis = 2)
        contour = np.concatenate([contour, contour, contour, contour, contour,contour], axis = 2)
        #print('New image shape 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',image.shape, contour.shape)

    slice_numr2 = int(np.ceil(contour.shape[2]/4))
    slice_numr3 = int(np.ceil(contour.shape[2]/2))
    slice_numr4 = int(np.ceil(3*contour.shape[2]/4))

    #print('slice_ second third ----------',contour.shape[2],slice_numr2,slice_numr3,slice_numr4)

    slice_num1 = np.random.choice(range(slice_numr2))
    slice_num2 = np.random.choice(range(slice_numr2,slice_numr3))
    slice_num3 = np.random.choice(range(slice_numr3,slice_numr4))
    slice_num4 = np.random.choice(range(slice_numr4,contour.shape[2]))
    #print('slice_num1,slice_num2,slice_num3,slice_num4----------',slice_num1,slice_num2,slice_num3,slice_num4)

    return image, contour, slice_num1, slice_num2, slice_num3, slice_num4


def format_train_image(image,contour, slice_num1, slice_num2, slice_num3, slice_num4, shape):

    norm_image1 = ct_train_normalize(image[:,:,slice_num1])
    ct_image1 = train_crop_spin_fun(norm_image1, contour, slice_num1, shape)


    norm_image2 = ct_train_normalize(image[:,:,slice_num2])
    ct_image2 = train_crop_spin_fun(norm_image2, contour, slice_num2, shape)

    norm_image3 = ct_train_normalize(image[:,:,slice_num3])
    ct_image3 = train_crop_spin_fun(norm_image3, contour, slice_num3, shape)


    norm_image4 = ct_train_normalize(image[:,:,slice_num4])
    ct_image4 = train_crop_spin_fun(norm_image4, contour, slice_num4, shape)

    # Image augmentation
    #pet_image = pet_augment(pet_image)
    ct_image1 = ct_augment(ct_image1)
    ct_image2 = ct_augment(ct_image2)
    ct_image3 = ct_augment(ct_image3)
    ct_image4 = ct_augment(ct_image4)

    return ct_image1, ct_image2, ct_image3, ct_image4


def train_crop_spin_fun(norm_image, contour, slice_num, shape):

    filled_contour = np.zeros((norm_image.shape))

    #for i in range(norm_img.shape[0]):
    it =  np.random.choice(range(4, 6))

    temp1 = binary_dilation(contour[:,:,slice_num],structure=np.ones((5,5)),iterations=it )
    temp2 = fill_contour(temp1)
    filled_contour[:,:,0] = temp2
    filled_contour[:,:,1] = temp2
    filled_contour[:,:,2] = temp2

    # bounding box coordinates
    c = bbox2(temp2)
    #print(c)

    # Segment out tumor from lung image
    full_tumor_image = np.where(filled_contour>0, norm_image, 0)
    #print(full_tumor_image.shape)
    tumor_image = full_tumor_image[c[0]:c[1],c[2]:c[3]]


    #print('before resize', tumor_image.shape)
    en_0 =  np.random.choice(range(tumor_image.shape[0], shape[0]))
    en_1 =  np.random.choice(range(tumor_image.shape[1], shape[1]))
    #print('random enlarge value ', en_0, en_1)
    tumor_image = resize(tumor_image, (en_0,en_1 ), mode='reflect')

    # fill the image
    #print('after resize', tumor_image.shape)
    fill_0 = shape[0]-tumor_image.shape[0]
    fill_1 = shape[1]-tumor_image.shape[1]
    #print('fill x and fill y',fill_0,fill_1)

    b_0 = np.random.choice(range(fill_0))
    f_0 = fill_0 - b_0

    b_1 = np.random.choice(range(fill_1))
    f_1 = fill_1 - b_1
    #print('b_0,f_0, b_1, f_1', b_0, f_0, b_1, f_1)
    tumor_image = np.pad(tumor_image, ((b_0,f_0),(b_1,f_1),(0,0)), 'constant', constant_values=(0, 0))
    #print('final size',tumor_image.shape)
    return tumor_image

def load_val_image(image_dataset_path, contour_dataset_path, shape):
    image = np.load(image_dataset_path.decode())
    contour = np.load(contour_dataset_path.decode())
    image = np.transpose(image, [1, 2, 0])
    contour = np.transpose(contour, [1, 2, 0])

    if contour.shape[2] == 3:
        #print('contour shape----------------------- ', contour.shape)
        image = np.concatenate([image, image], axis = 2)
        contour = np.concatenate([contour, contour], axis = 2)
        #print('New image shape 3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',image.shape, contour.shape)
    elif contour.shape[2] == 2:
        #print('contour shape----------------------- ', contour.shape)
        image = np.concatenate([image, image, image], axis = 2)
        contour = np.concatenate([contour, contour, contour], axis = 2)
        #print('New image shape 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',image.shape, contour.shape)

    elif contour.shape[2] == 1:
        #print('contour shape----------------------- ', contour.shape)
        image = np.concatenate([image, image, image,image,image, image], axis = 2)
        contour = np.concatenate([contour, contour, contour, contour, contour,contour], axis = 2)
        #print('New image shape 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',image.shape, contour.shape)


    slice_numr2 = int(np.ceil(contour.shape[2]/4))
    slice_numr3 = int(np.ceil(contour.shape[2]/2))
    slice_numr4 = int(np.ceil(3*contour.shape[2]/4))

    slice_num1 = 0
    slice_num2 = slice_numr2
    slice_num3 = slice_numr3
    slice_num4 = slice_numr4

    #print('slice_num1,slice_num2,slice_num3,slice_num4----------',slice_num1,slice_num2,slice_num3,slice_num4)
    #print('slice_ second third ----------',contour.shape[2],slice_numr2,slice_numr3,slice_numr4)

    return image, contour, slice_num1,slice_num2, slice_num3,slice_num4

def format_val_image(image,contour, slice_num1, slice_num2, slice_num3, slice_num4, shape):


    norm_image1 = ct_val_normalize(image[:,:,slice_num1])
    ct_image1 = val_crop_spin_fun(norm_image1, contour, slice_num1, shape)

    norm_image2 = ct_val_normalize(image[:,:,slice_num2])
    ct_image2 = val_crop_spin_fun(norm_image2, contour, slice_num2, shape)

    norm_image3 = ct_val_normalize(image[:,:,slice_num3])
    ct_image3 = val_crop_spin_fun(norm_image3, contour, slice_num3, shape)

    #image, contour, slice_num = load_train_image(image_ct_dataset_path, contour_ct_dataset_path,shape)
    norm_image4 = ct_val_normalize(image[:,:,slice_num4])
    ct_image4 = val_crop_spin_fun(norm_image4, contour, slice_num4, shape)

    # Image augmentation
    #pet_image = pet_augment(pet_image)
    ct_image1 = ct_augment(ct_image1)
    ct_image2 = ct_augment(ct_image2)
    ct_image3 = ct_augment(ct_image3)
    ct_image4 = ct_augment(ct_image4)

    return ct_image1, ct_image2, ct_image3, ct_image4

def val_crop_spin_fun(norm_image, contour, slice_num, shape):

    filled_contour = np.zeros((norm_image.shape))

    #for i in range(norm_img.shape[0]):
    it = np.random.choice(range(4,6))
    temp1 = binary_dilation(contour[:,:,slice_num],structure=np.ones((5,5)),iterations= it )
    temp2 = fill_contour(temp1)
    filled_contour[:,:,0] = temp2
    filled_contour[:,:,1] = temp2
    filled_contour[:,:,2] = temp2
    # Segment out tumor from lung image
    #tumor_image = np.where(filled_contour>0, norm_image, 0)
    #print(tumor_image.shape)
    c = bbox2(temp2)
    #print(c)

    # Segment out tumor from lung image
    full_tumor_image = np.where(filled_contour>0, norm_image, 0)
    #print(full_tumor_image.shape)
    tumor_image = full_tumor_image[c[0]:c[1],c[2]:c[3]]

    #print(tumor_image.shape)
    #print('before resize', tumor_image.shape)
    en_0 =  np.random.choice(range(tumor_image.shape[0], shape[0]))
    en_1 =  np.random.choice(range(tumor_image.shape[1], shape[1]))
    tumor_image = resize(tumor_image, (en_0,en_1 ), mode='reflect')

    # fill the image
    #print('after resize', tumor_image.shape)
    fill_0 = shape[0]-tumor_image.shape[0]
    fill_1 = shape[1]-tumor_image.shape[1]
    #print('fill x and fill y',fill_0,fill_1)

    b_0 = np.random.choice(range(fill_0))
    f_0 = fill_0 - b_0

    b_1 = np.random.choice(range(fill_1))
    f_1 = fill_1 - b_1

    tumor_image = np.pad(tumor_image, ((b_0,f_0),(b_1,f_1),(0,0)), 'constant', constant_values=(0, 0))
    #print('final size',tumor_image.shape)


    return tumor_image

def load_test_image(image_dataset_path, contour_dataset_path, shape):

    image = np.load(image_dataset_path.decode())
    contour = np.load(contour_dataset_path.decode())
    image = np.transpose(image, [1, 2, 0])
    contour = np.transpose(contour, [1, 2, 0])

    if contour.shape[2] == 3:
        #print('contour shape----------------------- ', contour.shape)
        image = np.concatenate([image, image], axis = 2)
        contour = np.concatenate([contour, contour], axis = 2)
        #print('New image shape 3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',image.shape, contour.shape)
    elif contour.shape[2] == 2:
        #print('contour shape----------------------- ', contour.shape)
        image = np.concatenate([image, image, image], axis = 2)
        contour = np.concatenate([contour, contour, contour], axis = 2)
        #print('New image shape 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',image.shape, contour.shape)

    elif contour.shape[2] == 1:
        #print('contour shape----------------------- ', contour.shape)
        image = np.concatenate([image, image, image,image,image, image], axis = 2)
        contour = np.concatenate([contour, contour, contour, contour, contour,contour], axis = 2)
        #print('New image shape 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',image.shape, contour.shape)

    slice_numr2 = int(np.ceil(contour.shape[2]/4))
    slice_numr3 = int(np.ceil(contour.shape[2]/2))
    slice_numr4 = int(np.ceil(3*contour.shape[2]/4))

    slice_num1 = 0
    slice_num2 = slice_numr2
    slice_num3 = slice_numr3
    slice_num4 = slice_numr4

    #print('slice_num1,slice_num2,slice_num3,slice_num4----------',slice_num1,slice_num2,slice_num3,slice_num4)
    #print('slice_ second third ----------',contour.shape[2],slice_numr2,slice_numr3,slice_numr4)


    return image, contour, slice_num1,slice_num2, slice_num3,slice_num4

def format_test_image(image,contour, slice_num1, slice_num2, slice_num3, slice_num4, shape):


    norm_image1 = ct_val_normalize(image[:,:,slice_num1])
    ct_image1 = test_crop_spin_fun(norm_image1, contour, slice_num1, shape)

    #image, contour, slice_num  = load_test_image(image_ct_dataset_path, contour_ct_dataset_path,shape)
    norm_image2 = ct_val_normalize(image[:,:,slice_num2])
    ct_image2 = test_crop_spin_fun(norm_image2, contour, slice_num2, shape)

    norm_image3 = ct_val_normalize(image[:,:,slice_num3])
    ct_image3 = test_crop_spin_fun(norm_image3, contour, slice_num3, shape)

    #image, contour, slice_num  = load_test_image(image_ct_dataset_path, contour_ct_dataset_path,shape)
    norm_image4 = ct_val_normalize(image[:,:,slice_num4])
    ct_image4 = test_crop_spin_fun(norm_image4, contour, slice_num4, shape)


    return ct_image1, ct_image2, ct_image3, ct_image4

def test_crop_spin_fun(norm_image, contour, slice_num, shape):

    filled_contour = np.zeros((norm_image.shape))

    #for i in range(norm_img.shape[0]):
    temp1 = binary_dilation(contour[:,:,slice_num],structure=np.ones((5,5)),iterations= 5 )
    temp2 = fill_contour(temp1)
    filled_contour[:,:,0] = temp2
    filled_contour[:,:,1] = temp2
    filled_contour[:,:,2] = temp2
    # Segment out tumor from lung image
    c = bbox2(temp2)
    #print(c)

    # Segment out tumor from lung image
    full_tumor_image = np.where(filled_contour>0, norm_image, 0)
    #print(full_tumor_image.shape)
    tumor_image = full_tumor_image[c[0]:c[1],c[2]:c[3]]

    #print(tumor_image.shape)
    #print('before resize', tumor_image.shape)
    en_0 =  299
    en_1 =  299
    tumor_image = resize(tumor_image, (en_0,en_1 ), mode='reflect')


    return tumor_image

def ct_augment(image):
    rot = np.random.choice(range(0,180))
    she1 = np.random.choice(range(1,30))
    she2 = np.random.choice(range(30,70))
    augment_img = iaa.Sequential([iaa.Affine(rotate=rot), iaa.OneOf([iaa.Affine(shear=(-she1, she1)) ,iaa.Flipud(0.5)]),iaa.OneOf([iaa.Fliplr(0.5),iaa.Affine(shear=(-she2, she2))])],random_order=True)
#,iaa.OneOf([iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)), iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))])
    image_aug = augment_img.augment_image(image)

    return image_aug

def pet_augment(image):
    rot = np.random.choice(range(0,180))
    she1 = np.random.choice(range(1,30))
    she2 = np.random.choice(range(30,70))
    augment_img = iaa.Sequential([iaa.Affine(rotate=rot), iaa.OneOf([iaa.Affine(shear=(-she1, she1)) ,iaa.Flipud(0.5)]),iaa.OneOf([iaa.Fliplr(0.5),iaa.Affine(shear=(-she2, she2))]),iaa.OneOf([iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)), iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))])],random_order=True)

    image_aug = augment_img.augment_image(image)

    return image_aug

def ct_train_normalize(image):
    norm_image = np.zeros((image.shape[0],image.shape[1],3))
    # zero mean each channel
    #image = image - np.mean(image)
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.


    # Normalized
    norm_image[:,:,0] = image

    # Contrast stretching
    low_lim =  np.random.choice(range(1,40))
    upper_lim =  np.random.choice(range(60,99))
    p2, p98 = np.percentile(image, (low_lim, upper_lim))
    norm_image[:,:,1] = exposure.rescale_intensity(image, in_range=(p2, p98))

    # Adaptive Equalization
    #norm_image[:,:,1] = exposure.equalize_adapthist(image, clip_limit=0.03)

    # Equalization
    norm_image[:,:,2] = exposure.equalize_hist(image)

    return norm_image

def pet_train_normalize(image):
    norm_image = np.zeros((image.shape[0],image.shape[1],3))
    # zero mean each channel
    #image = image - np.mean(image)
    MIN_BOUND = np.amin(image)
    MAX_BOUND = np.amax(image)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.


    # Normalized
    norm_image[:,:,0] = image

    # Contrast stretching
    low_lim =  np.random.choice(range(1,40))
    upper_lim =  np.random.choice(range(60,99))
    p2, p98 = np.percentile(image, (low_lim, upper_lim))
    norm_image[:,:,1] = exposure.rescale_intensity(image, in_range=(p2, p98))

    # Adaptive Equalization
    #norm_image[:,:,1] = exposure.equalize_adapthist(image, clip_limit=0.03)

    # Equalization
    norm_image[:,:,2] = exposure.equalize_hist(image)

    return norm_image

def ct_val_normalize(image):
    norm_image = np.zeros((image.shape[0],image.shape[1],3))
    # zero mean each channel
    #image = image - np.mean(image)
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.


    # Normalized
    norm_image[:,:,0] = image

    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    norm_image[:,:,1] = exposure.rescale_intensity(image, in_range=(p2, p98))

    # Adaptive Equalization
    #norm_image[:,:,1] = exposure.equalize_adapthist(image, clip_limit=0.03)

    # Equalization
    norm_image[:,:,2] = exposure.equalize_hist(image)

    return norm_image

def pet_val_normalize(image):
    norm_image = np.zeros((image.shape[0],image.shape[1],3))
    # zero mean each channel
    #image = image - np.mean(image)
    MIN_BOUND = np.amin(image)
    MAX_BOUND = np.amax(image)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    #print(image.shape)

    # Normalized
    norm_image[:,:,0] = image

    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    norm_image[:,:,1] = exposure.rescale_intensity(image, in_range=(p2, p98))

    # Adaptive Equalization
    #norm_image[:,:,1] = exposure.equalize_adapthist(image, clip_limit=0.03)

    # Equalization
    norm_image[:,:,2] = exposure.equalize_hist(image)

    return norm_image




def c_index3(month,risk, status):

    c_index = concordance_index(np.reshape(month, -1), -np.reshape(risk, -1), np.reshape(status, -1))

    return c_index





def encoder(CTpet_output1, CTpet_output2, CTpet_output3, CTpet_output4, keep_prob, is_training, hidden_layer):

    end_points = {}

    ct_enc1 = tf.reshape(CTpet_output1,[-1,CTpet_output1.shape[1]*CTpet_output1.shape[2]*CTpet_output1.shape[3]])
    ct_enc2 = tf.reshape(CTpet_output2,[-1,CTpet_output2.shape[1]*CTpet_output2.shape[2]*CTpet_output2.shape[3]])
    ct_enc3 = tf.reshape(CTpet_output3,[-1,CTpet_output3.shape[1]*CTpet_output3.shape[2]*CTpet_output3.shape[3]])
    ct_enc4 = tf.reshape(CTpet_output4,[-1,CTpet_output4.shape[1]*CTpet_output4.shape[2]*CTpet_output4.shape[3]])

    enc = tf.concat([ ct_enc1, ct_enc2, ct_enc3, ct_enc4 ], 1)
    print('Input layer shape >>>>>>>>>', enc.shape)

    # layer 0
    enc = tf.layers.batch_normalization( enc, axis=-1, renorm= False , training = is_training , name = 'batch_norm_layer_0')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_0'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_0')

    # layer 1
    enc = tf.layers.dense(enc, 4096, activation = None, kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_1')

    enc = tf.layers.batch_normalization( enc, axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_1')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_1'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_1')

    # layer 2
    enc = tf.layers.dense(enc, 2048, activation = None, kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_2')

    enc = tf.layers.batch_normalization( enc, axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_2')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_2'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_2')

    # layer 3
    enc = tf.layers.dense(enc, 1024, activation = None, kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_3')

    enc = tf.layers.batch_normalization( enc, axis=-1, renorm= False , training = is_training , name = 'batch_norm_layer_3')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_3'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_3')

    # layer 4
    enc = tf.layers.dense(enc, 512, activation = None, kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_4')

    enc = tf.layers.batch_normalization( enc, axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_4')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_4'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_4')

    # layer 5
    enc = tf.layers.dense(enc, 256, activation = None, kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_5')

    enc = tf.layers.batch_normalization( enc, axis=-1,  renorm= False , training = is_training, name = 'batch_norm_layer_5')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_5'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_5')

    # layer 6
    enc = tf.layers.dense(enc, 128, activation = None,  kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_6')

    enc = tf.layers.batch_normalization( enc, axis=-1,  renorm= False , training = is_training, name = 'batch_norm_layer_6')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_6'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_6')

    # layer 7
    enc = tf.layers.dense(enc, 64, activation = None, kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_7')

    enc = tf.layers.batch_normalization( enc, axis=-1,  renorm= False , training = is_training, name = 'batch_norm_layer_7')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_7'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_7')

    # layer 8
    enc = tf.layers.dense(enc, 32, activation = None,  kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_8')

    enc = tf.layers.batch_normalization( enc, axis=-1,  renorm= False , training = is_training, name = 'batch_norm_layer_8')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_8'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_8')

    # layer 9
    enc = tf.layers.dense(enc, 16, activation = None,  kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_9')

    enc = tf.layers.batch_normalization( enc, axis=-1,  renorm= False , training = is_training, name = 'batch_norm_layer_9')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_9'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_9')

    # layer 10
    enc = tf.layers.dense(enc, 8, activation = None,  kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_10')

    enc = tf.layers.batch_normalization( enc, axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_10')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_10'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_10')

    # layer 11
    enc = tf.layers.dense(enc, 4, activation = None,  kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_11')

    enc = tf.layers.batch_normalization( enc, axis=-1,  renorm= False , training = is_training, name = 'batch_norm_layer_11')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_11'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_11')

    # layer 12
    enc = tf.layers.dense(enc, 2, activation = None,  kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_12')

    enc = tf.layers.batch_normalization( enc, axis=-1,  renorm= False , training = is_training, name = 'batch_norm_layer_12')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_12'] = enc
    enc = tf.layers.dropout(enc, rate = keep_prob, training= is_training, name = 'dropout_layer_12')

    # layer 13
    enc = tf.layers.dense(enc, 1, activation = None ,  kernel_initializer= tf.contrib.layers.xavier_initializer(), name = 'enc_dense_layer_13')

    enc = tf.layers.batch_normalization( enc, axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_13')
    enc = tf.nn.leaky_relu(enc, alpha = 0.2)
    end_points['layer_13'] = enc
    print('Output layer shape >>>>>>>>>> ',end_points[hidden_layer].shape)



    return end_points

def negloglik_risk_function(affine, month, status):

    #risk ={}
    #nllk = {}
#for key, value in end_points.items():

    #feature: n*features, month: n*1, status: n*1
    batch_size = tf.shape(affine)[0]

    print('shape of features',affine.shape)
    #affine = value
    #affine = tf.matmul(feature, parameters["cox_beta"]) #n*1

    affine = tf.reduce_sum(affine, axis= 1)
    affine = tf.expand_dims(affine, -1)
    #mean, variance = tf.nn.moments(affine, axes= 0)
    affine = tf.divide(affine - tf.reduce_mean(affine), tf.reduce_max(affine)-tf.reduce_min(affine))

    print('shape after norm',affine.shape)
    risk= tf.exp(affine)
    print('shape of risk',risk.shape)

        # risk set matrix with columns denoting each event risk set
    mask = tf.less_equal(tf.transpose(month), tf.tile(month, [1, batch_size])) #n*n

    # matrix denoting events that occure at the same time
    s_mask = tf.equal(tf.transpose(month), tf.tile(month, [1, batch_size])) # n*n

    # sum  risk over all patients that are in each event risk set
    sum_risk = tf.matmul(tf.transpose(risk), tf.to_float(mask)) #1*n

    # sum over log-risk (affine) over all patients that have equal event time
    sum_affine = tf.matmul(tf.transpose(affine), tf.to_float(s_mask))  #1*n
    # log of risk summation
    log_risk_sum = tf.log(sum_risk+ 1e-8 ) #+ np.exp(1)
    # number of patient that have equal time
    d = tf.reduce_sum(tf.cast(s_mask, tf.float32), axis= 0)

    # Multiply each patient risk with corresponding d value
    risk_mul_d = tf.multiply(log_risk_sum, d)

    status = tf.cast(status,tf.float32)

    # calculate likelyhood at each time event
    nllk = -tf.multiply(tf.add( sum_affine, -risk_mul_d ),tf.transpose(status))

    # sum over all time events

    nllk= tf.reduce_sum(nllk)


    return risk, nllk, d





def hidden_layer_fun(hidden_layer_size):
    # Death

    if hidden_layer_size == 1:
        hidden_layer = 'layer_13'
        print('hidden_layer 12 with 1 features')
    elif hidden_layer_size == 2:
        hidden_layer = 'layer_12'
        print('hidden_layer 11 with 2 features')
    elif hidden_layer_size == 4:
        hidden_layer = 'layer_11'
        print('hidden_layer 10 with 4 features')
    elif hidden_layer_size == 8:
        hidden_layer = 'layer_10'
        print('hidden_layer 9 with 8 features')
    elif hidden_layer_size == 16:
        hidden_layer = 'layer_9'
        print('hidden_layer 8 with 16 features')
    elif hidden_layer_size == 32:
        hidden_layer = 'layer_8'
        print('hidden_layer 7 with 32 features')
    elif hidden_layer_size == 64:
        hidden_layer = 'layer_7'
        print('hidden_layer 8 with 64 features')
    else:
        print('hidden_layer check')
    return hidden_layer

def event_fun(event):
    # Death
    if event == 'OS':
        time_index = 0
        status_index = 1
    if event == 'RFP':
        time_index = 2
        status_index = 3
    if event == 'DC':
        time_index = 4
        status_index = 5
    if event == 'RC':
        time_index = 10
        status_index = 11
    if event == 'DSS':
        time_index = 0
        status_index = 12
    return time_index, status_index


# OS ==>  DEATH
# RFP ==> Event or Rec
# DC ==> Dist
# DSS ==> CAUSE
# RC ==> Reg


time_index,status_index = event_fun('OS')

hidden_layer_size = 1
hidden_layer = hidden_layer_fun(hidden_layer_size)

learning_rate = 0.001
num_epochs = 30000
minibatch_size = 90
keep_prob = 0.8

lam = 0.001

# K Cross val suffle and split
k_fold_split = 3
k_fold_num = 0

end_points ={}



num_sample_per_fold = int(len(full_ct_image_dataset_path)/k_fold_split)

templ = np.arange(1,len(full_ct_image_dataset_path)+1)
open(output_file1, 'w+').close()
for k_fold_num in range(k_fold_split):
    f1 = open(output_file1, 'a')
    print('k_fold_num ----------------------------',k_fold_num)
    f1.write('k_fold_num ----------------------------'  + str(k_fold_num) + '\n')

    print('number of k foldes',k_fold_split)

    print(templ-1,full_ct_image_dataset_path)
    f1.write('Full sample file list ' + '\n' + str(full_ct_image_dataset_path) + '\n')
    f1.write('Corresponding sample file indices ' + '\n' + str(templ-1) + '\n')

    ## Create mask vector, set initialy all elements to False mask.
    masked_event_label = np.ma.array(templ, mask=False)

    ## Create censored/uncensored sample index vector.
    event_label = full_dataset_labels[:,(time_index,status_index)] # Target event label

    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y] # Get index function
    censored_event_index = get_indexes(0,event_label[:,1])
    uncensored_events_index = get_indexes(1,event_label[:,1])

    ## Validation fold sample selection. Set condtion for equal number of censored/uncensored samples in validation fold.
    sample_size = int(num_sample_per_fold/2)
    np.random.seed(13 +k_fold_num)  # set random seed diff for each fold (not sure if we actually need this)
    rand_censored_indices= np.random.choice(censored_event_index ,sample_size, replace= False )
    rand_uncensored_indices= np.random.choice(uncensored_events_index , sample_size, replace= False )

    ## Mask validation fold
    masked_event_label.mask[rand_censored_indices] = True
    masked_event_label.mask[rand_uncensored_indices] = True

    ## Select training fold's samples
    train_indices = masked_event_label.nonzero()
    print('Train_indices',train_indices[0])
    f1.write('Train_indices'+ '\n' + str(train_indices) + '\n')
    f1.write('Training set sample list' + '\n' +  str(full_ct_image_dataset_path[train_indices]) + '\n')

    #test_censored_indices = rand_censored_indices
    #test_uncensored_indices = rand_uncensored_indices
    #test_indices = np.concatenate((test_censored_indices, test_uncensored_indices))
    test_indices = np.concatenate((rand_censored_indices, rand_uncensored_indices))
    print('Test_indices',test_indices)
    f1.write('Test_indices' + '\n' + str(test_indices) + '\n')
    f1.write('test set sample list' + '\n' + str(full_ct_image_dataset_path[test_indices]) + '\n')
    f1.close()


    ##### Reset to defualt graph every k-fold validation iteration
    trainable_weights_graph = tf.Graph()
    with trainable_weights_graph.as_default():
        with tf.name_scope('Data_Input_Pipline'):

            train_dataset = tf.data.Dataset.from_tensor_slices((full_ct_image_dataset_path[train_indices],full_ct_contour_dataset_path[train_indices], full_highres_ct_image_dataset_path[train_indices],full_highres_ct_contour_dataset_path[train_indices],full_post_ct_image_dataset_path[train_indices],full_post_ct_contour_dataset_path[train_indices],full_dataset_features[train_indices],full_dataset_labels[train_indices]))

            test_dataset = tf.data.Dataset.from_tensor_slices((full_ct_image_dataset_path[test_indices],full_ct_contour_dataset_path[test_indices],full_dataset_features[test_indices],full_dataset_labels[test_indices]))

            train_size = len(train_indices[0])
            val_size = len(test_indices)
            test_size = len(test_indices)

            print('size of training validiation and test sets ---------------',train_size, val_size, test_size)
            n_batch = math.ceil(train_size / minibatch_size)
            print('Number of batches ',n_batch)

            train_dataset = train_dataset.shuffle(buffer_size = minibatch_size).prefetch( buffer_size = minibatch_size )
            val_dataset = test_dataset.shuffle(buffer_size= val_size).prefetch(buffer_size= val_size)
            test_dataset = test_dataset.shuffle(buffer_size= test_size).prefetch(buffer_size= test_size)
    #
            train_dataset = train_dataset.map((lambda image_ct_dataset_path, contour_ct_dataset_path,image_highres_ct_dataset_path, contour_highres_ct_dataset_path,image_post_ct_dataset_path, contour_post_ct_dataset_path, dataset_features, dataset_labels: tf.py_func(create_train, [image_ct_dataset_path, contour_ct_dataset_path,image_highres_ct_dataset_path, contour_highres_ct_dataset_path,image_post_ct_dataset_path, contour_post_ct_dataset_path, dataset_features, dataset_labels], [np.float32,np.float32,np.float32,np.float32,np.float32,np.float32,np.float32,np.float32, np.float32,np.float32,np.float32,np.float32, np.float32, np.int16])), num_parallel_calls= 8).batch(minibatch_size).repeat()

            val_dataset = val_dataset.map((lambda image_ct_dataset_path, contour_ct_dataset_path, dataset_features, dataset_labels: tf.py_func(create_val, [image_ct_dataset_path, contour_ct_dataset_path, dataset_features, dataset_labels], [np.float32,np.float32,np.float32,np.float32, np.float32, np.int16])), num_parallel_calls= 8).batch(batch_size = val_size).repeat()

            test_dataset = test_dataset.map((lambda image_ct_dataset_path, contour_ct_dataset_path, dataset_features, dataset_labels: tf.py_func(create_test, [image_ct_dataset_path, contour_ct_dataset_path, dataset_features, dataset_labels], [np.float32,np.float32,np.float32,np.float32, np.float32, np.int16])), num_parallel_calls= 8).batch(batch_size = test_size).repeat()

            train_iterator = train_dataset.make_initializable_iterator()
            val_iterator = val_dataset.make_initializable_iterator()
            test_iterator = test_dataset.make_initializable_iterator()

            train_next_element = train_iterator.get_next()
            val_next_element = val_iterator.get_next()
            test_next_element = test_iterator.get_next()




        ##### Create place holder
        with tf.name_scope("inputs") as scope:
            X1 = tf.placeholder(tf.float32, shape = (None, 299, 299, 3), name = 'CTpet1_input_full_Image')
            X2 = tf.placeholder(tf.float32, shape = (None, 299, 299, 3), name = 'CTpet2_input_full_Image')
            X3 = tf.placeholder(tf.float32, shape = (None, 299, 299, 3), name = 'CTpet3_input_full_Image')
            X4 = tf.placeholder(tf.float32, shape = (None, 299, 299, 3), name = 'CTpet4_input_full_Image')

            Z = tf.placeholder(tf.float32, shape = (None, 5), name = 'input_features')
            L = tf.placeholder(tf.int32, shape = (None, 13), name = 'labels')

            tf_is_training = tf.placeholder(tf.bool, name = 'tf_is_training')
            tf_learning_rate = tf.placeholder_with_default(0.0001, shape=None, name="tf_learning_rate")


        ## Create main CNN pipeline
        with tf.device('/device:GPU:0'):
            with tf.variable_scope(None, 'CTpet_output1'):
                with slim.arg_scope(inception_v4.inception_v4_arg_scope()):

                    _, CTpet_end_points1 = inception_v4.inception_v4(X1 , is_training = tf_is_training, dropout_keep_prob= keep_prob )
                    CTpet_output1 = CTpet_end_points1['global_pool']


        with tf.device('/device:GPU:0'):
            with tf.variable_scope(None, 'CTpet_output2'):
                with slim.arg_scope(inception_v4.inception_v4_arg_scope()):#

                    _, CTpet_end_points2 = inception_v4.inception_v4(X2 , is_training = tf_is_training, dropout_keep_prob= keep_prob)
                    CTpet_output2 = CTpet_end_points2['global_pool']


        with tf.device('/device:GPU:1'):
            with tf.variable_scope(None, 'CTpet_output3'):
                with slim.arg_scope(inception_v4.inception_v4_arg_scope()):#

                    _, CTpet_end_points3 = inception_v4.inception_v4(X3 , is_training = tf_is_training, dropout_keep_prob= keep_prob)
                    CTpet_output3 = CTpet_end_points3['global_pool']


        with tf.device('/device:GPU:1'):
            with tf.variable_scope(None, 'CTpet_output4'):
                with slim.arg_scope(inception_v4.inception_v4_arg_scope()):#

                    _, CTpet_end_points4 = inception_v4.inception_v4(X4 , is_training = tf_is_training, dropout_keep_prob= keep_prob)
                    CTpet_output4 = CTpet_end_points4['global_pool']


            # Encode pictures
            # Create the model, use the default arg scope to configure the batch norm parameters.


        tf.summary.image('CTpet1_Input image', tf.expand_dims(X1[:,:,:,0],-1), max_outputs=1)
        tf.summary.image('CTpet2_Input image', tf.expand_dims(X2[:,:,:,0],-1), max_outputs=1)
        tf.summary.image('CTpet3_Input image', tf.expand_dims(X3[:,:,:,0],-1), max_outputs=1)
        tf.summary.image('CTpet4_Input image', tf.expand_dims(X4[:,:,:,0],-1), max_outputs=1)
        tf.summary.histogram('CTpet1_output',CTpet_output1)
        tf.summary.histogram('CTpet2_output',CTpet_output2)
        tf.summary.histogram('CTpet3_output',CTpet_output3)
        tf.summary.histogram('CTpet4_output',CTpet_output4)


        # FC encoder block
        #arg_scope = tf.contrib.framework.arg_scope
        #with arg_scope([tf.layers.dense, tf.layers.batch_normalization], reuse= tf.AUTO_REUSE):
        with tf.device('/device:GPU:0'):
            with tf.variable_scope('FC_layers'):
                end_points  = encoder(CTpet_output1, CTpet_output2, CTpet_output3, CTpet_output4,  keep_prob,  tf_is_training, hidden_layer)


        # CoxPH negative loglikelyhood function
        #with tf.device("/cpu:1"):
        with tf.name_scope('CoxPH_negloglik'):

            month = tf.expand_dims(L[:, time_index], 1)
            status = tf.expand_dims(L[:, status_index], 1)

            ## Negative loglikelyhood function
            risk, nllk, d = negloglik_risk_function(end_points[hidden_layer], month, status)

        #tf.summary.histogram('encoder_output',encoder_output)
        #tf.summary.histogram('risk',risk[hidden_layer])
        #tf.summary.scalar('nllk cost',nllk[hidden_layer])

        for key, value in end_points.items():
            tf.summary.histogram('encoder_output '+str(key) ,value)

        #for key, value in risk.items():
        #    tf.summary.histogram('risk ' +str(key) ,value)



        with tf.name_scope('Conv_layers_regularization'):
            l2_loss = lam * (sum(tf.nn.l2_loss(var) for var in tf.trainable_variables() if (not 'bias' in var.name)))

        tf.summary.scalar('l2_loss', l2_loss)


        ## Total loss function
        with tf.name_scope('Reg_loss_function'):
            total_loss = nllk + l2_loss


        ## Trainable parameters
        train_var1 = [var for var in tf.trainable_variables() if  (not 'CTpet_output2/' in var.name) and  (not 'CTpet_output3/' in var.name) and  (not 'CTpet_output4/' in var.name) and (not 'FC_layers' in var.name) and (not 'CTpet_output1/InceptionV4/Logits/Logits' in var.name) and  (not 'CTpet_output1/InceptionV4/AuxLogits' in var.name) and (not 'CTpet_output1/InceptionV4/Conv2d_1a_3x3' in var.name) and (not 'CTpet_output1/InceptionV4/Conv2d_2a_3x3' in var.name) and (not 'CTpet_output1/InceptionV4/Conv2d_2b_3x3' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_3' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_4' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_5' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_6' in var.name)]# and (not 'CTpet_output1/InceptionV4/Mixed_6b' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_6c' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_6d' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_6e' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_6f' in var.name)]

        train_var11 = [var for var in tf.trainable_variables() if  (not 'CTpet_output2/' in var.name) and  (not 'CTpet_output3/' in var.name) and  (not 'CTpet_output4/' in var.name) and (not 'FC_layers' in var.name) and (not 'CTpet_output1/InceptionV4/Logits/Logits' in var.name) and  (not 'CTpet_output1/InceptionV4/AuxLogits' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_7' in var.name) and (not 'CTpet_output1/InceptionV4/Conv2d_1a_3x3' in var.name) and (not 'CTpet_output1/InceptionV4/Conv2d_2a_3x3' in var.name) and (not 'CTpet_output1/InceptionV4/Conv2d_2b_3x3' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_3' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_4' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_5' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_6a' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_6b' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_6c' in var.name) and (not 'CTpet_output1/InceptionV4/Mixed_6d' in var.name)]


        train_var2 = [var for var in tf.trainable_variables() if (not 'CTpet_output1/' in var.name) and  (not 'CTpet_output3/' in var.name) and  (not 'CTpet_output4/' in var.name) and (not 'FC_layers' in var.name) and (not 'CTpet_output2/InceptionV4/Logits/Logits' in var.name) and  (not 'CTpet_output2/InceptionV4/AuxLogits' in var.name) and (not 'CTpet_output2/InceptionV4/Conv2d_1a_3x3' in var.name) and (not 'CTpet_output2/InceptionV4/Conv2d_2a_3x3' in var.name) and (not 'CTpet_output2/InceptionV4/Conv2d_2b_3x3' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_3' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_4' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_5' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_6' in var.name)]# and (not 'CTpet_output2/InceptionV4/Mixed_6b' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_6c' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_6d' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_6e' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_6f' in var.name)]

        train_var22 = [var for var in tf.trainable_variables() if (not 'CTpet_output1/' in var.name) and  (not 'CTpet_output3/' in var.name) and  (not 'CTpet_output4/' in var.name) and (not 'FC_layers' in var.name) and (not 'CTpet_output2/InceptionV4/Logits/Logits' in var.name) and  (not 'CTpet_output2/InceptionV4/AuxLogits' in var.name)  and (not 'CTpet_output2/InceptionV4/Mixed_7' in var.name) and (not 'CTpet_output2/InceptionV4/Conv2d_1a_3x3' in var.name) and (not 'CTpet_output2/InceptionV4/Conv2d_2a_3x3' in var.name) and (not 'CTpet_output2/InceptionV4/Conv2d_2b_3x3' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_3' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_4' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_5' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_6a' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_6b' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_6c' in var.name) and (not 'CTpet_output2/InceptionV4/Mixed_6d' in var.name)]


        train_var3 = [var for var in tf.trainable_variables() if  (not 'CTpet_output1/' in var.name) and  (not 'CTpet_output2/' in var.name) and  (not 'CTpet_output4/' in var.name) and (not 'FC_layers' in var.name) and (not 'CTpet_output3/InceptionV4/Logits/Logits' in var.name) and  (not 'CTpet_output3/InceptionV4/AuxLogits' in var.name) and (not 'CTpet_output3/InceptionV4/Conv2d_1a_3x3' in var.name) and (not 'CTpet_output3/InceptionV4/Conv2d_2a_3x3' in var.name) and (not 'CTpet_output3/InceptionV4/Conv2d_2b_3x3' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_3' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_4' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_5' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_6' in var.name)]# and (not 'CTpet_output3/InceptionV4/Mixed_6b' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_6c' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_6d' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_6e' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_6f' in var.name)]

        train_var33 = [var for var in tf.trainable_variables() if  (not 'CTpet_output1/' in var.name) and  (not 'CTpet_output2/' in var.name) and  (not 'CTpet_output4/' in var.name) and (not 'FC_layers' in var.name) and (not 'CTpet_output3/InceptionV4/Logits/Logits' in var.name) and  (not 'CTpet_output3/InceptionV4/AuxLogits' in var.name)  and (not 'CTpet_output3/InceptionV4/Mixed_7' in var.name) and (not 'CTpet_output3/InceptionV4/Conv2d_1a_3x3' in var.name) and (not 'CTpet_output3/InceptionV4/Conv2d_2a_3x3' in var.name) and (not 'CTpet_output3/InceptionV4/Conv2d_2b_3x3' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_3' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_4' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_5' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_6a' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_6b' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_6c' in var.name) and (not 'CTpet_output3/InceptionV4/Mixed_6d' in var.name)]


        train_var4 = [var for var in tf.trainable_variables() if (not 'CTpet_output1/' in var.name) and  (not 'CTpet_output2/' in var.name) and  (not 'CTpet_output3/' in var.name) and (not 'FC_layers' in var.name) and (not 'CTpet_output4/InceptionV4/Logits/Logits' in var.name) and  (not 'CTpet_output4/InceptionV4/AuxLogits' in var.name) and (not 'CTpet_output4/InceptionV4/Conv2d_1a_3x3' in var.name) and (not 'CTpet_output4/InceptionV4/Conv2d_2a_3x3' in var.name) and (not 'CTpet_output4/InceptionV4/Conv2d_2b_3x3' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_3' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_4' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_5' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_6' in var.name)]# and (not 'CTpet_output4/InceptionV4/Mixed_6b' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_6c' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_6d' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_6e' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_6f' in var.name)]

        train_var44 = [var for var in tf.trainable_variables() if (not 'CTpet_output1/' in var.name) and  (not 'CTpet_output2/' in var.name) and  (not 'CTpet_output3/' in var.name) and (not 'FC_layers' in var.name) and (not 'CTpet_output4/InceptionV4/Logits/Logits' in var.name) and  (not 'CTpet_output4/InceptionV4/AuxLogits' in var.name)  and (not 'CTpet_output4/InceptionV4/Mixed_7' in var.name) and (not 'CTpet_output4/InceptionV4/Conv2d_1a_3x3' in var.name) and (not 'CTpet_output4/InceptionV4/Conv2d_2a_3x3' in var.name) and (not 'CTpet_output4/InceptionV4/Conv2d_2b_3x3' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_3' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_4' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_5' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_6a' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_6b' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_6c' in var.name) and (not 'CTpet_output4/InceptionV4/Mixed_6d' in var.name)]



        train_var_fc = [var for var in tf.trainable_variables() if (not 'CTpet_output1/' in var.name) and  (not 'CTpet_output2/' in var.name) and  (not 'CTpet_output3/' in var.name) and  (not 'CTpet_output4/' in var.name)]

        #

        for var in train_var1:
            print('Training parameters 1-------------------------------',var)
            tf.summary.histogram(str(var.name),var)

        for var in train_var11:
            print('Training parameters 11-------------------------------',var)
            tf.summary.histogram(str(var.name),var)

        for var in train_var2:
            print('Training parameters 2-------------------------------',var)
            tf.summary.histogram(str(var.name),var)

        for var in train_var22:
            print('Training parameters 22-------------------------------',var)
            tf.summary.histogram(str(var.name),var)

        for var in train_var3:
            print('Training parameters 3-------------------------------',var)
            tf.summary.histogram(str(var.name),var)

        for var in train_var33:
            print('Training parameters 33-------------------------------',var)
            tf.summary.histogram(str(var.name),var)


        for var in train_var4:
            print('Training parameters 4-------------------------------',var)
            tf.summary.histogram(str(var.name),var)

        for var in train_var44:
            print('Training parameters 44-------------------------------',var)
            tf.summary.histogram(str(var.name),var)

        for var in train_var_fc:
            print('Training parameters fc-------------------------------',var)
            tf.summary.histogram(str(var.name),var)




        # OPTIMIZATION - Also updates batchnorm operations automatically
        with tf.device('/device:GPU:0'):
            with tf.variable_scope('CTpet_output_opt1') as scope:
                tf_optimizer1 = tf.train.AdamOptimizer(tf_learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CTpet_output1')
                with tf.control_dependencies(update_ops):
                    optimize1 = tf_optimizer1.minimize(total_loss,var_list = train_var1, name="train_op1")

        with tf.device('/device:GPU:2'):
            with tf.variable_scope('CTpet_output_opt1') as scope:
                tf_optimizer11 = tf.train.AdamOptimizer(tf_learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CTpet_output1')
                with tf.control_dependencies(update_ops):
                    optimize11 = tf_optimizer11.minimize(total_loss ,var_list = train_var11, name="train_op1")


        with tf.device('/device:GPU:0'):
            with tf.variable_scope('CTpet_output_opt2') as scope:
                tf_optimizer2 = tf.train.AdamOptimizer(tf_learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CTpet_output2')
                with tf.control_dependencies(update_ops):
                    optimize2 = tf_optimizer2.minimize(total_loss, var_list = train_var2, name="train_op2")

        with tf.device('/device:GPU:2'):
            with tf.variable_scope('CTpet_output_opt2') as scope:
                tf_optimizer22 = tf.train.AdamOptimizer(tf_learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CTpet_output2')
                with tf.control_dependencies(update_ops):
                    optimize22 = tf_optimizer22.minimize(total_loss ,var_list = train_var22, name="train_op2")


        # OPTIMIZATION - Also updates batchnorm operations automatically
        with tf.device('/device:GPU:1'):
            with tf.variable_scope('CTpet_output_opt3') as scope:
                tf_optimizer3 = tf.train.AdamOptimizer(tf_learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CTpet_output3')
                with tf.control_dependencies(update_ops):
                    optimize3 = tf_optimizer3.minimize(total_loss ,var_list = train_var3, name="train_op3")

        with tf.device('/device:GPU:3'):
            with tf.variable_scope('CTpet_output_opt3') as scope:
                tf_optimizer33 = tf.train.AdamOptimizer(tf_learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CTpet_output3')
                with tf.control_dependencies(update_ops):
                    optimize33 = tf_optimizer33.minimize(total_loss ,var_list = train_var33, name="train_op3")


        with tf.device('/device:GPU:1'):
            with tf.variable_scope('CTpet_output_opt4') as scope: #
                tf_optimizer4 = tf.train.AdamOptimizer(tf_learning_rate) #, name= 'opt4'
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CTpet_output4')
                with tf.control_dependencies(update_ops):
                    optimize4 = tf_optimizer4.minimize(total_loss ,var_list = train_var4, name="train_op4")

        with tf.device('/device:GPU:3'):
            with tf.variable_scope('CTpet_output_opt4') as scope: #
                tf_optimizer44 = tf.train.AdamOptimizer(tf_learning_rate) #, name= 'opt4'
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'CTpet_output4')
                with tf.control_dependencies(update_ops):
                    optimize44 = tf_optimizer44.minimize(total_loss ,var_list = train_var44, name="train_op4")


        with tf.device('/device:GPU:1'):
            with tf.variable_scope('FC_layers_opt') as scope:
                tf_optimizer_fc = tf.train.AdamOptimizer(tf_learning_rate) # , name= 'fc_opt'
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'FC_layers')
                with tf.control_dependencies(update_ops):
                    optimize_fc = tf_optimizer_fc.minimize(total_loss ,var_list = train_var_fc, name="fc_train_op")


        # PRETRAINED SAVER SETTINGS
        # Lists of scopes of weights to include/exclude from pretrained snapshot

        CTpet1_pretrained_include = ['CTpet_output1/InceptionV4/']
        CTpet1_pretrained_exclude = ['CTpet_output1/InceptionV4/AuxLogits', 'CTpet_output1/InceptionV4/Logits/Logits', 'CTpet_output1/InceptionV4/Mixed_7' ,'CTpet_output1/InceptionV4/Mixed_6h','CTpet_output1/InceptionV4/Mixed_6g']

        CTpet1_pretrained_vars = tf.contrib.framework.get_variables_to_restore(
            include = CTpet1_pretrained_include,
            exclude = CTpet1_pretrained_exclude)

        CTpet2_pretrained_include = ['CTpet_output2/InceptionV4/']
        CTpet2_pretrained_exclude = [ 'CTpet_output2/InceptionV4/AuxLogits', 'CTpet_output2/InceptionV4/Logits/Logits', 'CTpet_output2/InceptionV4/Mixed_7' ,'CTpet_output2/InceptionV4/Mixed_6h','CTpet_output2/InceptionV4/Mixed_6g']

        CTpet2_pretrained_vars = tf.contrib.framework.get_variables_to_restore(
            include = CTpet2_pretrained_include,
            exclude = CTpet2_pretrained_exclude)

        CTpet3_pretrained_include = ['CTpet_output3/InceptionV4/']
        CTpet3_pretrained_exclude = [ 'CTpet_output3/InceptionV4/AuxLogits', 'CTpet_output3/InceptionV4/Logits/Logits', 'CTpet_output3/InceptionV4/Mixed_7' ,'CTpet_output3/InceptionV4/Mixed_6h','CTpet_output3/InceptionV4/Mixed_6g']

        CTpet3_pretrained_vars = tf.contrib.framework.get_variables_to_restore(
            include = CTpet3_pretrained_include,
            exclude = CTpet3_pretrained_exclude)

        CTpet4_pretrained_include = ['CTpet_output4/InceptionV4/']
        CTpet4_pretrained_exclude = [ 'CTpet_output4/InceptionV4/AuxLogits', 'CTpet_output4/InceptionV4/Logits/Logits', 'CTpet_output4/InceptionV4/Mixed_7' ,'CTpet_output4/InceptionV4/Mixed_6h','CTpet_output4/InceptionV4/Mixed_6g' ]

        CTpet4_pretrained_vars = tf.contrib.framework.get_variables_to_restore(
            include = CTpet4_pretrained_include,
            exclude = CTpet4_pretrained_exclude)





        pretrained_include = ['CTpet_output1/','CTpet_output2/','CTpet_output3/','CTpet_output4/','FC_layers/']
        pretrained_exclude = ['CTpet_output1/InceptionV4/AuxLogits', 'CTpet_output1/InceptionV4/Logits/Logits','CTpet_output2/InceptionV4/AuxLogits', 'CTpet_output2/InceptionV4/Logits/Logits','CTpet_output3/InceptionV4/AuxLogits', 'CTpet_output3/InceptionV4/Logits/Logits','CTpet_output4/InceptionV4/AuxLogits', 'CTpet_output4/InceptionV4/Logits/Logits']

        # PRETRAINED SAVER - For loading pretrained weights on the first run

        pretrained_vars = tf.contrib.framework.get_variables_to_restore(
            include = pretrained_include,
            exclude = pretrained_exclude)


        CTpet1_pretrained_vars = {v.name.lstrip('CTpet_output1/').rstrip(':0'): v for v in CTpet1_pretrained_vars}
        CTpet2_pretrained_vars = {v.name.lstrip('CTpet_output2/').rstrip(':0'): v for v in CTpet2_pretrained_vars}
        CTpet3_pretrained_vars = {v.name.lstrip('CTpet_output3/').rstrip(':0'): v for v in CTpet3_pretrained_vars}
        CTpet4_pretrained_vars = {v.name.lstrip('CTpet_output4/').rstrip(':0'): v for v in CTpet4_pretrained_vars}

        #PETct_pretrained_vars = {v.name.lstrip('PETct_output/').rstrip(':0'): v for v in PETct_pretrained_vars}
        print('CTpet1_pretrained_vars-----------------------------------------------------',CTpet1_pretrained_vars)
        #print('CTpet2_pretrained_vars-----------------------------------------------------',CTpet2_pretrained_vars)

        # Restore from pretrained model
        CTpet1_tf_imagenet_pretrained_saver = tf.train.Saver(CTpet1_pretrained_vars, name="CTpet1_imagenet_pretrained_saver")
        CTpet2_tf_imagenet_pretrained_saver = tf.train.Saver(CTpet2_pretrained_vars, name="CTpet2_imagenet_pretrained_saver")
        CTpet3_tf_imagenet_pretrained_saver = tf.train.Saver(CTpet3_pretrained_vars, name="CTpet3_imagenet_pretrained_saver")
        CTpet4_tf_imagenet_pretrained_saver = tf.train.Saver(CTpet4_pretrained_vars, name="CTpet4_imagenet_pretrained_saver")

        # PRETRAINED SAVER SETTINGS
        # Lists of scopes of weights to include/exclude from pretrained snapshot

        CTpet1_include = ['CTpet_output1/InceptionV4/']
        CTpet1_exclude = ['CTpet_output1/InceptionV4/AuxLogits', 'CTpet_output1/InceptionV4/Logits/Logits','CTpet_output1/InceptionV4/Conv2d_1a_3x3' ,'CTpet_output1/InceptionV4/Conv2d_2a_3x3' , 'CTpet_output1/InceptionV4/Conv2d_2b_3x3' , 'CTpet_output1/InceptionV4/Mixed_3', 'CTpet_output1/InceptionV4/Mixed_4' , 'CTpet_output1/InceptionV4/Mixed_5' ,'CTpet_output1/InceptionV4/Mixed_6a', 'CTpet_output1/InceptionV4/Mixed_6b','CTpet_output1/InceptionV4/Mixed_6c' ,'CTpet_output1/InceptionV4/Mixed_6d' ,'CTpet_output1/InceptionV4/Mixed_6e','CTpet_output1/InceptionV4/Mixed_6f']

        CTpet1_vars = tf.contrib.framework.get_variables_to_restore(
            include = CTpet1_include,
            exclude = CTpet1_exclude)

        CTpet2_include = ['CTpet_output2/InceptionV4/']
        CTpet2_exclude = [ 'CTpet_output2/InceptionV4/AuxLogits', 'CTpet_output2/InceptionV4/Logits/Logits', 'CTpet_output2/InceptionV4/Conv2d_1a_3x3' ,'CTpet_output2/InceptionV4/Conv2d_2a_3x3' , 'CTpet_output2/InceptionV4/Conv2d_2b_3x3' , 'CTpet_output2/InceptionV4/Mixed_3', 'CTpet_output2/InceptionV4/Mixed_4' , 'CTpet_output2/InceptionV4/Mixed_5' ,'CTpet_output2/InceptionV4/Mixed_6a','CTpet_output2/InceptionV4/Mixed_6b','CTpet_output2/InceptionV4/Mixed_6c','CTpet_output2/InceptionV4/Mixed_6d','CTpet_output2/InceptionV4/Mixed_6e','CTpet_output2/InceptionV4/Mixed_6f']

        CTpet2_vars = tf.contrib.framework.get_variables_to_restore(
            include = CTpet2_include,
            exclude = CTpet2_exclude)

        CTpet3_include = ['CTpet_output3/InceptionV4/']
        CTpet3_exclude = [ 'CTpet_output3/InceptionV4/AuxLogits', 'CTpet_output3/InceptionV4/Logits/Logits', 'CTpet_output3/InceptionV4/Conv2d_1a_3x3' ,'CTpet_output3/InceptionV4/Conv2d_2a_3x3' , 'CTpet_output3/InceptionV4/Conv2d_2b_3x3' , 'CTpet_output3/InceptionV4/Mixed_3', 'CTpet_output3/InceptionV4/Mixed_4' , 'CTpet_output3/InceptionV4/Mixed_5' ,'CTpet_output3/InceptionV4/Mixed_6a','CTpet_output3/InceptionV4/Mixed_6b','CTpet_output3/InceptionV4/Mixed_6c','CTpet_output3/InceptionV4/Mixed_6d','CTpet_output3/InceptionV4/Mixed_6e','CTpet_output3/InceptionV4/Mixed_6f']

        CTpet3_vars = tf.contrib.framework.get_variables_to_restore(
            include = CTpet3_include,
            exclude = CTpet3_exclude)

        CTpet4_include = ['CTpet_output4/InceptionV4/']
        CTpet4_exclude = [ 'CTpet_output4/InceptionV4/AuxLogits', 'CTpet_output4/InceptionV4/Logits/Logits','CTpet_output4/InceptionV4/Conv2d_1a_3x3' ,'CTpet_output4/InceptionV4/Conv2d_2a_3x3' , 'CTpet_output4/InceptionV4/Conv2d_2b_3x3' , 'CTpet_output4/InceptionV4/Mixed_3', 'CTpet_output4/InceptionV4/Mixed_4' , 'CTpet_output4/InceptionV4/Mixed_5' ,'CTpet_output4/InceptionV4/Mixed_6a','CTpet_output4/InceptionV4/Mixed_6b','CTpet_output4/InceptionV4/Mixed_6c','CTpet_output4/InceptionV4/Mixed_6d','CTpet_output4/InceptionV4/Mixed_6e','CTpet_output4/InceptionV4/Mixed_6f']
        CTpet4_vars = tf.contrib.framework.get_variables_to_restore(
            include = CTpet4_include,
            exclude = CTpet4_exclude)

        CTpet_fc_include = ['FC_layers/']
        CTpet_fc_exclude = []

        CTpet_fc_vars = tf.contrib.framework.get_variables_to_restore(
            include = CTpet_fc_include,
            exclude = CTpet_fc_exclude)

        print('CTpet3_vars-----------------------------------------------------',CTpet3_vars)
        print('CTpet_fc_vars-----------------------------------------------------',CTpet_fc_vars)

        # Main checkpoint SAVER - For saving/restoring your complete model
        CTpet1_tf_saver = tf.train.Saver(CTpet1_vars,name="CTpet1_saver")
        CTpet2_tf_saver = tf.train.Saver(CTpet2_vars,name="CTpet2_saver")
        CTpet3_tf_saver = tf.train.Saver(CTpet3_vars,name="CTpet3_saver")
        CTpet4_tf_saver = tf.train.Saver(CTpet4_vars,name="CTpet4_saver")
        CTpet_fc_tf_saver = tf.train.Saver(CTpet_fc_vars,name="CTpet_fc_saver")

        #pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")
        tf_saver = tf.train.Saver(name="saver")



        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.gpu_options.allow_growth = True

        with tf.Session(config = config) as sess:
        #with tf.Session() as sess:
            sess.run(init)
            initialize_vars(session=sess)
            CTpet1_initialize_vars(session=sess)
            CTpet2_initialize_vars(session=sess)
            CTpet3_initialize_vars(session=sess)
            CTpet4_initialize_vars(session=sess)
            CTpet_fc_initialize_vars(session=sess)
            #PETct_initialize_vars(session=sess)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(log_output + '/train' + str(k_fold_num), sess.graph)
            val_writer = tf.summary.FileWriter(log_output + '/val' + str(k_fold_num))
            test_writer = tf.summary.FileWriter(log_output + '/test' + str(k_fold_num))

            sess.run(train_iterator.initializer)
            sess.run(val_iterator.initializer)
            sess.run(test_iterator.initializer)

            open(output_file, 'w+').close()

            for epoch in range(num_epochs):
                f = open(output_file, 'a')

                cum_minibatch_loss = 0.0
                cum_minibatch_cost = 0.0
                cum_minibatch_c_indices = 0.0


                for batch in range(n_batch):
                    #batch_images, batch_features , batch_labels = sess.run(training_next_element)
                    ct_batch_images1, ct_batch_images2, ct_batch_images3, ct_batch_images4, highres_ct_batch_images1, highres_ct_batch_images2, highres_ct_batch_images3, highres_ct_batch_images4,post_ct_batch_images1, post_ct_batch_images2, post_ct_batch_images3, post_ct_batch_images4, feature_dat, label_dat = sess.run(train_next_element)
                    #
                    # Post CT training
                    _,_,_,_,_,_,_,_,_, train_risk, train_total_loss, train_nllk_cost, d_list,  train_status, train_month = sess.run([optimize_fc,optimize1,optimize2,optimize3,optimize4,optimize11,optimize22,optimize33,optimize44, risk, total_loss, nllk, d, status, month], feed_dict={X1: post_ct_batch_images1, X2: post_ct_batch_images2, X3: post_ct_batch_images3, X4: post_ct_batch_images4, L: label_dat, tf_is_training: True, tf_learning_rate: learning_rate}) #Y: PETct_img_dat,
                    print(d_list, train_total_loss/minibatch_size)

                    # Highres CT training
                    _,_,_,_,_,_,_,_,_, train_risk, train_total_loss, train_nllk_cost, d_list,  train_status, train_month = sess.run([optimize_fc,optimize1,optimize2,optimize3,optimize4,optimize11,optimize22,optimize33,optimize44, risk, total_loss, nllk, d, status, month], feed_dict={X1: highres_ct_batch_images1, X2: highres_ct_batch_images2, X3: highres_ct_batch_images3, X4: highres_ct_batch_images4, L: label_dat, tf_is_training: True, tf_learning_rate: learning_rate}) #Y: PETct_img_dat,
                    print(d_list, train_total_loss/minibatch_size)

                    # PRE_CT training
                    _,_,_,_,_,_,_,_,_, train_risk, train_total_loss, train_nllk_cost, d_list,  train_status, train_month, summary_train = sess.run([optimize_fc,optimize1,optimize2,optimize3,optimize4,optimize11,optimize22,optimize33,optimize44, risk, total_loss, nllk, d, status, month, merged], feed_dict={X1: ct_batch_images1, X2: ct_batch_images2, X3: ct_batch_images3, X4: ct_batch_images4, L: label_dat, tf_is_training: True, tf_learning_rate: learning_rate}) #Y: PETct_img_dat,

                    print(d_list, train_total_loss/minibatch_size)

                    print('batch risk score shape for batch number and epoch number----:',train_risk.shape, batch, epoch)
                    #print('batch loss ',train_total_loss[hidden_layer]/minibatch_size)

                    #for key, value in train_total_loss.items():
                    cum_minibatch_loss += train_total_loss
                    cum_minibatch_cost += train_nllk_cost
                    train_c_indices = c_index3(train_month, train_risk,train_status)
                    cum_minibatch_c_indices += train_c_indices


                epoch_loss = cum_minibatch_loss / (n_batch *minibatch_size)
                epoch_cost = cum_minibatch_cost / (n_batch *minibatch_size)
                epoch_c_indices = cum_minibatch_c_indices / n_batch


                print('Training loss after epoch', epoch, ':', epoch_loss)
                f.write('Training loss after epoch ' + str(epoch) + ': ' + str(epoch_loss) + '\n')
                print('C-index for training set after epoch', epoch, ': ', epoch_c_indices)
                f.write('C-index for training set after epoch' + str(epoch) + ': ' + str(epoch_c_indices) + '\n')

                if (epoch+1)% 10 == 0:
                    train_writer.add_summary(summary_train, epoch)
                    summary = tf.Summary()

                    summary.value.add(tag='lambda', simple_value= lam)
                    summary.value.add(tag='hidden_layer_size', simple_value= hidden_layer_size)
                    summary.value.add(tag='learning_rate', simple_value= learning_rate)
                    summary.value.add(tag='keep_prob', simple_value= keep_prob)
                    summary.value.add(tag='minibatch_size', simple_value= minibatch_size)

                    #for key, value in train_c_indices.items():

                    summary.value.add(tag='train total loss ' +str(hidden_layer) , simple_value= epoch_loss)
                    summary.value.add(tag='train nllk cost ' +str(hidden_layer) , simple_value= epoch_cost)
                    summary.value.add(tag='train c_index ' +str(hidden_layer) , simple_value= epoch_c_indices)

                    # Add to summary
                    train_writer.add_summary(summary, epoch)


                if (epoch+1)% 20 == 0:
                    # Evaluation
                    print('-------------Validation----------------')

                    CTpet1_img_dat,CTpet2_img_dat,CTpet3_img_dat,CTpet4_img_dat, feature_dat, label_dat = sess.run(val_next_element)
                    if epoch+1 < 40:
                        val_nllk_cost, val_total_loss, val_month, val_status, val_risk = sess.run([ nllk, total_loss, month, status, risk], feed_dict = {X1: CTpet1_img_dat, X2: CTpet2_img_dat, X3: CTpet3_img_dat, X4: CTpet4_img_dat,  Z: feature_dat, L: label_dat, tf_is_training: False})

                        val_epoch_loss = val_total_loss / val_size
                        val_epoch_cost = val_nllk_cost / val_size

                        print('val nllk cost after epoch', epoch, ':', val_epoch_cost)
                        f.write('val nllk cost after epoch ' + str(epoch) + ': ' + str(val_epoch_cost) + '\n')
                        print('val total loss after epoch', epoch, ':', val_epoch_loss)
                        f.write('val total loss after epoch ' + str(epoch) + ': ' + str(val_epoch_loss) + '\n')


                        val_c_indices = c_index3(val_month, val_risk ,val_status)
                        print('val risk',val_risk)
                        print('C-index for validation set after epoch', epoch, ': ', val_c_indices)
                        f.write('C-index for validation set after epoch' + str(epoch) + ': ' + str(val_c_indices) + '\n')

                    if epoch+1 > 39:
                        val_summary, val_nllk_cost, val_total_loss, val_month, val_status, val_risk = sess.run([merged, nllk, total_loss, month, status, risk], feed_dict = {X1: CTpet1_img_dat, X2: CTpet2_img_dat, X3: CTpet3_img_dat, X4: CTpet4_img_dat,  Z: feature_dat, L: label_dat, tf_is_training: False})

                        val_writer.add_summary(val_summary, epoch)


                        val_epoch_loss = val_total_loss / val_size
                        val_epoch_cost = val_nllk_cost / val_size

                        print('val nllk cost after epoch', epoch, ':', val_epoch_cost)
                        f.write('val nllk cost after epoch ' + str(epoch) + ': ' + str(val_epoch_cost) + '\n')
                        print('val total loss after epoch', epoch, ':', val_epoch_loss)
                        f.write('val total loss after epoch ' + str(epoch) + ': ' + str(val_epoch_loss) + '\n')

                        # validatin c-index

                        summary = tf.Summary()
                        #for key, value in val_risk.items():

                        val_c_indices = c_index3(val_month, val_risk ,val_status)
                        summary.value.add(tag='validation c_index ' +str(hidden_layer), simple_value= val_c_indices)
                        summary.value.add(tag='validation nllk cost ' +str(hidden_layer), simple_value= val_epoch_cost)
                        summary.value.add(tag='validation total loss ' +str(hidden_layer), simple_value= val_epoch_loss)

                        val_writer.add_summary(summary, epoch)

                        print('val risk',val_risk)
                        print('C-index for validation set after epoch', epoch, ': ', val_c_indices)
                        f.write('C-index for validation set after epoch' + str(epoch) + ': ' + str(val_c_indices) + '\n')

                    print('-------------Test----------------')

                    CTpet1_img_dat,CTpet2_img_dat,CTpet3_img_dat,CTpet4_img_dat, feature_dat, label_dat = sess.run(test_next_element)

                    if epoch+1 < 40:
                        test_nllk_cost, test_total_loss, test_month, test_status, test_risk = sess.run([ nllk, total_loss, month, status, risk], feed_dict = {X1: CTpet1_img_dat, X2: CTpet2_img_dat, X3: CTpet3_img_dat, X4: CTpet4_img_dat, Z: feature_dat, L: label_dat,  tf_is_training: False})

                        test_epoch_loss = test_total_loss / val_size
                        test_epoch_cost = test_nllk_cost / val_size

                        print('test nllk cost after epoch', epoch, ':', test_epoch_cost)
                        f.write('test nllk cost after epoch ' + str(epoch) + ': ' + str(test_epoch_cost) + '\n')
                        print('test total loss after epoch', epoch, ':', test_epoch_loss)
                        f.write('test total loss after epoch ' + str(epoch) + ': ' + str(test_epoch_loss) + '\n')

                        test_c_indices = c_index3(test_month, test_risk ,test_status)
                        print('test risk',test_risk)
                        print('C-index for test set after epoch', epoch, ': ', test_c_indices)
                        f.write('C-index for test set after epoch' + str(epoch) + ': ' + str(test_c_indices) + '\n')

                    if epoch+1 > 39:

                        test_summary, test_nllk_cost, test_total_loss, test_month, test_status, test_risk = sess.run([merged, nllk, total_loss, month, status, risk], feed_dict = {X1: CTpet1_img_dat, X2: CTpet2_img_dat, X3: CTpet3_img_dat, X4: CTpet4_img_dat, Z: feature_dat, L: label_dat,  tf_is_training: False})

                        test_writer.add_summary(test_summary, epoch)

                        test_epoch_loss = test_total_loss / val_size
                        test_epoch_cost = test_nllk_cost / val_size

                        print('test nllk cost after epoch', epoch, ':', test_epoch_cost)
                        f.write('test nllk cost after epoch ' + str(epoch) + ': ' + str(test_epoch_cost) + '\n')
                        print('test total loss after epoch', epoch, ':', test_epoch_loss)
                        f.write('test total loss after epoch ' + str(epoch) + ': ' + str(test_epoch_loss) + '\n')

                        summary = tf.Summary()

                        #for key, value in test_risk.items():
                        test_c_indices = c_index3(test_month, test_risk ,test_status)
                        summary.value.add(tag='test_c_index ' +str(hidden_layer), simple_value= test_c_indices)
                        summary.value.add(tag='test total loss ' +str(hidden_layer), simple_value= test_epoch_loss)
                        summary.value.add(tag='test nllk cost ' +str(hidden_layer), simple_value= test_epoch_cost)
                        test_writer.add_summary(summary, epoch)

                        print('test risk',test_risk)
                        print('C-index for test set after epoch', epoch, ': ', test_c_indices)
                        f.write('C-index for test set after epoch' + str(epoch) + ': ' + str(test_c_indices) + '\n')

                    # SAVE SNAPSHOT - after each epoch
                    print('save model')
                    tf_saver.save(sess, SNAPSHOT_FILE + '_'+str(k_fold_num) )
                    f.close()
