'''
initialize to create landmarks combined with CACD img
image_list = glob.glob("../carc/CACD2000/*.jpg")
save_path=with h5py.File("./data/CACD2000_{}.hdf5".format(name), 'w') as f:
create short subdataset with length=SHORT_LEN=15000
'''
SHORT_LEN=15000
import face_alignment
import glob
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
import random
import h5py

random.seed(2)
print ("Set random seed to 2!")
#There may be some corrupted image returns Nonetype in CACD, just skip them
block_img_list = ["24_Alan_Ritchson_0006.jpg", "50_Suzie_Plakson_0012.jpg", "18_Ross_Bagley_0005.jpg", "38_Maggie_Siff_0011.jpg", "58_Tony_Todd_0011.jpg", "37_Sanaa_Lathan_0007.jpg", "34_Robin_Thicke_0008.jpg", "56_Tress_MacNeille_0010.jpg", "29_Devon_Sawa_0006.jpg", "52_Roberto_Benigni_0004.jpg", "42_Michelle_Yeoh_0011.jpg", "54_Hulk_Hogan_0005.jpg", "43_Michelle_Yeoh_0009.jpg", "55_Bill_Mumy_0012.jpg", "57_Mark_Boone_Junior_0010.jpg", "56_Jim_Cummings_0015.jpg", "43_Shane_Black_0006.jpg", "48_Amy_Sedaris_0014.jpg", "43_Aidan_Gillen_0011.jpg", "48_Brad_Bird_0008.jpg", "53_Catherine_Bach_0005.jpg", "47_Brad_Bird_0005.jpg", "39_Naveen_Andrews_0014.jpg", "34_Burn_Gorman_0006.jpg", "19_Mae_Whitman_0009.jpg", "26_Jaimie_Alexander_0006.jpg", "56_Sting_0004.jpg", "33_Linda_Cardellini_0004.jpg", "58_Robert_Picardo_0007.jpg"]

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

image_list = glob.glob("../carc/CACD2000/*.jpg")
random.shuffle(image_list)
num_image = len(image_list)

#Create subdataset with SHORT_LEN
num_image=SHORT_LEN
image_list=image_list[:num_image]

num_train_image = int(num_image*0.70)
num_val_image = int(num_image*0.15)
num_test_image = num_image-num_train_image-num_val_image
print('The length of train,val,test=',num_train_image,num_val_image,num_test_image)
def prepare_dataset(name, size, _image_list):

    with h5py.File("./subdata/CACD2000_{}.hdf5".format(name), 'w') as f:
        image_dataset = f.create_dataset("img", shape=(size, 250, 250 , 3), dtype='uint8')
        landmark_dataset = f.create_dataset("lmk_2D", shape=(size, 68, 2), dtype='uint8')
        for idx, image_path in tqdm(enumerate(_image_list), desc="Processing Landmark...", total=size):
            if image_path.split('/')[-1] in block_img_list:
                continue
            image = io.imread(image_path)
            if image is None:
                print('Nonetype occurs')
                continue
            image_dataset[idx] = image
            landmark = fa.get_landmarks(image)
            if landmark is None:
                print ("Error in {}".format(image_path))
            else:
                landmark_dataset[idx] = landmark[0][:,:2]

print ("Prepare train set:")
prepare_dataset("train", num_train_image, image_list[:num_train_image])
print ("Prepare validation set:")
prepare_dataset("val", num_val_image, image_list[num_train_image:num_train_image+num_val_image])
print ("Prepare test set:")
prepare_dataset("test", num_test_image, image_list[-num_test_image])
