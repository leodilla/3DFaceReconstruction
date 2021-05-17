'''
set NUM_EPOCH=25, BATCH_SIZE=8
Load dataset with train and val dataloader in ./data
    val_dataset = CACDDataset("./data/CACD2000_val.hdf5")
    train_dataset = CACDDataset("./data/CACD2000_train.hdf5")
Save the visualize result of each epoch in "./result"
Save the model trained of each epoch in"./model_trained/"
Save the training loss curve in ('train.png')
'''
import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from tqdm import tqdm
from random import shuffle
import cv2
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat, savemat
from array import array
from skimage.io import imsave

import trimesh
import soft_renderer as sr
import face_alignment
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections

from facenet_pytorch import InceptionResnetV1
import h5py




# -------------------------- Hyperparameter ------------------------------
# Specify number of epochs, batch size and learning rate
NUM_EPOCH=25     # e.g. 40
VERBOSE_STEP=50 
BATCH_SIZE = 8   # e.g. 8
VISUAL_IDX=3     #random test to visualize
SEED=0

# -------------------------- Reproducibility ------------------------------
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Use the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # cuda:0

# -------------------------- Prepossing data in CACDDataset ------------------------------


train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

val_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

inv_normalize = transforms.Compose([
                    transforms.Normalize(
                                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                std=[1/0.229, 1/0.224, 1/0.255])
    ])


class CACDDataset(Dataset):

    def __init__(self, dataset_path, transforms, inv_normalize, residual_path=None):
        super(CACDDataset, self).__init__()
        self.dataset_path = dataset_path
        with h5py.File(dataset_path, 'r') as file:
            self.length = len(file['img'])
        self.transforms = transforms
        self.inv_normalize = inv_normalize
        self.residual_path = residual_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, "r") as file:
            img = file['img'][idx]
            landmark = file['lmk_2D'][idx]
        input_img = self.transforms(img)
        target_img = self.inv_normalize(input_img)

        if self.residual_path is not None:
            with h5py.File(self.residual_path, 'r') as file:
                recon_img = file['bfm_recon'][idx]
                recon_param = file['bfm_param'][idx]
            recon_img = self.transforms(recon_img[:, :, :3])
            return input_img, target_img, landmark, recon_img, recon_param
        else:
            return input_img, target_img, landmark


val_dataset = CACDDataset("./data/CACD2000_val.hdf5", val_transform, inv_normalize)
train_dataset = CACDDataset("./data/CACD2000_train.hdf5", train_transform, inv_normalize)

print ("Validation set real size: {}".format(len(val_dataset)))
print ("Training set real size: {}".format(len(train_dataset)))

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=0)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                      num_workers=0)


#Costruct resnet50 network
class BaseModel(nn.Module):
    """Get the base network, which is modified from ResNet50
        modify structure of FC layer
    """
    def __init__(self, IF_PRETRAINED=False):
        super(BaseModel, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=IF_PRETRAINED)
        self.resnet50.fc = nn.Linear(2048, 258) #the output params

    def forward(self, images):
        return self.resnet50(images)


class BFM_torch(nn.Module):
    """
    This is a torch implementation of the BFM model
    Used in the DNN model, comes with gradient support
    """

    def __init__(self):
        super(BFM_torch, self).__init__()
        model_path = './BFM/BFM_model_front.mat'
        model = loadmat(model_path)
        # [107127, 1]
        self.register_buffer("meanshape", torch.tensor(model['meanshape'].T, dtype=torch.float32))
        # [107127, 80]
        self.register_buffer("idBase", torch.tensor(model['idBase'], dtype=torch.float32))
        # [107127, 64]
        self.register_buffer("exBase", torch.tensor(model['exBase'], dtype=torch.float32))
        # [107127, 1]
        self.register_buffer("meantex", torch.tensor(model['meantex'].T, dtype=torch.float32))
        # [107121, 80]
        self.register_buffer('texBase', torch.tensor(model['texBase'], dtype=torch.float32))
        # [70789, 3]
        self.register_buffer('tri', torch.tensor(model['tri'], dtype=torch.int32))
        # [35709, 8] Max is 70789;
        self.register_buffer('point_buf', torch.tensor(model['point_buf'], dtype=torch.int32))
        # [68]
        self.register_buffer('keypoints',
                             torch.tensor(np.squeeze(model['keypoints']).astype(np.int32) - 1, dtype=torch.int32))

    def get_shape(self, id_param, ex_param):
        """
        Perform shape assembly from index parameter and expression parameter
        id_param: [bs, 80]
        ex_param: [bs, 64]
        return: [bs, 107127, 1]
        """
        assert id_param.shape[0] == ex_param.shape[0]
        bs = id_param.shape[0]

        id_base = self.idBase[None, :, :].expand(bs, -1, -1)
        ex_base = self.exBase[None, :, :].expand(bs, -1, -1)

        face_shape = self.meanshape + torch.bmm(id_base, id_param[:, :, None]) + torch.bmm(ex_base,
                                                                                           ex_param[:, :, None])
        face_shape = face_shape.reshape(bs, -1, 3)
        face_shape = face_shape - torch.mean(self.meanshape[None, :, :].reshape(1, -1, 3), dim=1, keepdim=True)
        return face_shape

    def get_texture(self, tex_param):
        """
        Perform texture assembly from texture parameter
        tex_param: [bs, 80]
        return: [bs, 107127, 1]
        """
        bs = tex_param.shape[0]
        tex_base = self.texBase[None, :, :].expand(bs, -1, -1)

        return self.meantex + torch.bmm(tex_base, tex_param[:, :, None])

    def compute_rotation_matrix(self, rotate_param):
        """
        Perform rotation based on the batch rotation parameter
        rotate_param: [bs, 3]
        return: [bs, 3, 3]
        """
        pitch, yaw, roll = rotate_param[:, 0], rotate_param[:, 1], rotate_param[:, 2]
        bs = rotate_param.shape[0]
        device = rotate_param.device

        pitch_matrix = torch.eye(3, device=device)[None, :, :].expand(bs, -1, -1).clone()
        yaw_matrix = torch.eye(3, device=device)[None, :, :].expand(bs, -1, -1).clone()
        roll_matrix = torch.eye(3, device=device)[None, :, :].expand(bs, -1, -1).clone()

        pitch_matrix[:, 1, 1] = torch.cos(pitch)
        pitch_matrix[:, 2, 2] = torch.cos(pitch)
        pitch_matrix[:, 1, 2] = -torch.sin(pitch)
        pitch_matrix[:, 2, 1] = torch.sin(pitch)

        yaw_matrix[:, 0, 0] = torch.cos(yaw)
        yaw_matrix[:, 2, 2] = torch.cos(yaw)
        yaw_matrix[:, 0, 2] = torch.sin(yaw)
        yaw_matrix[:, 2, 0] = -torch.sin(yaw)

        roll_matrix[:, 0, 0] = torch.cos(roll)
        roll_matrix[:, 1, 1] = torch.cos(roll)
        roll_matrix[:, 0, 1] = -torch.sin(roll)
        roll_matrix[:, 1, 0] = torch.sin(roll)

        return torch.bmm(torch.bmm(roll_matrix, yaw_matrix), pitch_matrix).permute(0, 2, 1)

class BFMFaceLoss(nn.Module):
    """Decode from the learned parameters to the 3D face model"""

    def __init__(self, renderer, device):
        super(BFMFaceLoss, self).__init__()
        self.BFM_model = BFM_torch().to(device)
        self.renderer = renderer

        self.mse_criterion = nn.MSELoss()
        self.sl1_criterion = nn.SmoothL1Loss()
        self.device = device

        self.a0 = torch.tensor(math.pi).to(self.device)
        self.a1 = torch.tensor(2 * math.pi / math.sqrt(3.0)).to(self.device)
        self.a2 = torch.tensor(2 * math.pi / math.sqrt(8.0)).to(self.device)
        self.c0 = torch.tensor(1 / math.sqrt(4 * math.pi)).to(self.device)
        self.c1 = torch.tensor(math.sqrt(3.0) / math.sqrt(4 * math.pi)).to(self.device)
        self.c2 = torch.tensor(3 * math.sqrt(5.0) / math.sqrt(12 * math.pi)).to(self.device)

        self.reverse_z = torch.eye(3).to(self.device)[None, :, :]
        self.face_net = InceptionResnetV1(pretrained='vggface2').eval()
        for param in self.face_net.parameters():
            param.requires_grad = False
        self.face_net.to(device)

    def split(self, params):
        id_coef = params[:, :80]
        ex_coef = params[:, 80:144]
        tex_coef = params[:, 144:224]
        angles = params[:, 224:227]
        gamma = params[:, 227:254]
        translation = params[:, 254:257]
        scale = params[:, 257:]
        return id_coef, ex_coef, tex_coef, angles, gamma, translation, scale

    def compute_norm(self, vertices):
        """
        Compute the norm of the vertices
        Input:
            vertices[bs, 35709, 3]
        """
        bs = vertices.shape[0]
        face_id = torch.flip(self.BFM_model.tri.reshape(-1, 3) - 1, dims=[1])
        point_id = self.BFM_model.point_buf - 1
        # [bs, 70789, 3]
        face_id = face_id[None, :, :].expand(bs, -1, -1)
        # [bs, 35709, 8]
        point_id = point_id[None, :, :].expand(bs, -1, -1)
        # [bs, 70789, 3] Gather the vertex location
        v1 = torch.gather(vertices, dim=1, index=face_id[:, :, :1].expand(-1, -1, 3).long())
        v2 = torch.gather(vertices, dim=1, index=face_id[:, :, 1:2].expand(-1, -1, 3).long())
        v3 = torch.gather(vertices, dim=1, index=face_id[:, :, 2:].expand(-1, -1, 3).long())
        # Compute the edge
        e1 = v1 - v2
        e2 = v2 - v3
        # Normal [bs, 70789, 3]
        norm = torch.cross(e1, e2)
        # Normal appended with zero vector [bs, 70790, 3]
        norm = torch.cat([norm, torch.zeros(bs, 1, 3).to(self.device)], dim=1)
        # [bs, 35709*8, 3]
        point_id = point_id.reshape(bs, -1)[:, :, None].expand(-1, -1, 3)
        # [bs, 35709*8, 3]
        v_norm = torch.gather(norm, dim=1, index=point_id.long())
        v_norm = v_norm.reshape(bs, 35709, 8, 3)
        # [bs, 35709, 3]
        v_norm = F.normalize(torch.sum(v_norm, dim=2), dim=-1)
        return v_norm

    def lighting(self, norm, albedo, gamma):
        """
        Add lighting to the albedo surface
        gamma: [bs, 27]
        norm: [bs, num_vertex, 3]
        albedo: [bs, num_vertex, 3]
        """
        assert norm.shape[0] == albedo.shape[0]
        assert norm.shape[0] == gamma.shape[0]
        bs = gamma.shape[0]
        num_vertex = norm.shape[1]

        init_light = torch.zeros(9).to(self.device)
        init_light[0] = 0.8
        gamma = gamma.reshape(bs, 3, 9) + init_light

        Y0 = self.a0 * self.c0 * torch.ones(bs, num_vertex, 1, device=self.device)
        Y1 = -self.a1 * self.c1 * norm[:, :, 1:2]
        Y2 = self.a1 * self.c1 * norm[:, :, 2:3]
        Y3 = -self.a1 * self.c1 * norm[:, :, 0:1]
        Y4 = self.a2 * self.c2 * norm[:, :, 0:1] * norm[:, :, 1:2]
        Y5 = -self.a2 * self.c2 * norm[:, :, 1:2] * norm[:, :, 2:3]
        Y6 = self.a2 * self.c2 * 0.5 / math.sqrt(3.0) * (3 * norm[:, :, 2:3] ** 2 - 1)
        Y7 = -self.a2 * self.c2 * norm[:, :, 0:1] * norm[:, :, 2:3]
        Y8 = self.a2 * self.c2 * 0.5 * (norm[:, :, 0:1] ** 2 - norm[:, :, 1:2] ** 2)
        # [bs, num_vertice, 9]
        Y = torch.cat([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8], dim=2)

        light_color = torch.bmm(Y, gamma.permute(0, 2, 1))
        vertex_color = light_color * albedo
        return vertex_color

    def reconst_img(self, params, return_type=None):
        bs = params.shape[0]
        id_coef, ex_coef, tex_coef, angles, gamma, tranlation, scale = self.split(params)

        face_shape = self.BFM_model.get_shape(id_coef, ex_coef)
        face_albedo = self.BFM_model.get_texture(tex_coef)
        face_shape[:, :, -1] *= -1
        # Recenter the face mesh
        face_albedo = face_albedo.reshape(bs, -1, 3) / 255.

        # face model scaling, rotation and translation
        rotation_matrix = self.BFM_model.compute_rotation_matrix(angles)
        face_shape = torch.bmm(face_shape, rotation_matrix)
        # Compute the normal
        normal = self.compute_norm(face_shape)

        face_shape = (1 + scale[:, :, None]) * face_shape
        face_shape = face_shape + tranlation[:, None, :]

        face_albedo = self.lighting(normal, face_albedo, gamma)

        tri = torch.flip(self.BFM_model.tri.reshape(-1, 3) - 1, dims=[-1])
        face_triangles = tri[None, :, :].expand(bs, -1, -1)

        #recon_mesh, recon_img = self.renderer(face_shape,face_triangles,face_albedo,texture_type="vertex")
        recon_img = self.renderer(face_shape, face_triangles, face_albedo, texture_type="vertex")
        if return_type == 'all':
            return recon_img, face_shape, face_triangles, face_albedo
        else:
            return recon_img

    def forward(self, params, gt_img, gt_lmk):
        bs = params.shape[0]
        id_coef, ex_coef, tex_coef, angles, gamma, tranlation, scale = self.split(params)

        face_shape = self.BFM_model.get_shape(id_coef, ex_coef)
        face_albedo = self.BFM_model.get_texture(tex_coef)
        face_shape[:, :, -1] *= -1
        # Recenter the face mesh
        face_albedo = face_albedo.reshape(bs, -1, 3) / 255.

        # face model scaling, rotation and translation
        rotation_matrix = self.BFM_model.compute_rotation_matrix(angles)
        face_shape = torch.bmm(face_shape, rotation_matrix)
        # Compute the normal
        normal = self.compute_norm(face_shape)

        face_shape = (1 + scale[:, :, None]) * face_shape
        face_shape = face_shape + tranlation[:, None, :]

        face_albedo = self.lighting(normal, face_albedo, gamma)

        tri = torch.flip(self.BFM_model.tri.reshape(-1, 3) - 1, dims=[-1])
        face_triangles = tri[None, :, :].expand(bs, -1, -1)

        #recon_mesh, recon_img = self.renderer(face_shape,face_triangles,face_albedo,texture_type="vertex")
        recon_img = self.renderer(face_shape, face_triangles, face_albedo, texture_type="vertex")
        recon_mesh = sr.Mesh(face_shape, face_triangles, face_albedo, texture_type="vertex")
        recon_lmk = recon_mesh.vertices[:, self.BFM_model.keypoints.long(), :]

        # Compute loss
        # remove the alpha channel
        mask = (recon_img[:, -1:, :, :].detach() > 0).float()
        # Image loss
        img_loss = self.mse_criterion(recon_img[:, :3, :, :], gt_img * mask)
        # Landmark loss 
        recon_lmk_2D_rev = (recon_lmk[:, :, :2] + 1) * 250./ 2.
        recon_lmk_2D = (recon_lmk[:, :, :2] + 1) * 250. / 2.
        recon_lmk_2D[:, :, 1] = 250. - recon_lmk_2D_rev[:, :, 1]
        lmk_loss = self.sl1_criterion(recon_lmk_2D, gt_lmk.float())
        # face recog loss
        recon_feature = self.face_net(recon_img[:, :3, :, :])
        gt_feature = self.face_net(gt_img * mask)
        recog_loss = self.mse_criterion(recon_feature, gt_feature)
        all_loss = img_loss + lmk_loss + 10 * recog_loss
        return all_loss, img_loss, lmk_loss, recog_loss, recon_img



# -------------------------- Model loading ------------------------------
model = BaseModel(IF_PRETRAINED=True)
model.to(device)

# -------------------------- Optimizer loading --------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
lr_schduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5)


# ------------------------- Loss loading --------------------------------
camera_distance = 2.732
elevation = 0
azimuth = 0

renderer = sr.SoftRenderer(image_size=224, sigma_val=1e-4, aggr_func_rgb='hard',
                            camera_mode='look_at', viewing_angle=30, fill_back=False,
                            perspective=False, light_intensity_ambient=1.0, light_intensity_directionals=0)

renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
face_loss = BFMFaceLoss(renderer, device)

#visualize batch
# ------------------------- plot visualization --------------------------
def visualize_batch(gt_imgs, recon_imgs):
    gt_imgs = gt_imgs.cpu()
    recon_imgs = recon_imgs.cpu()
    bs = gt_imgs.shape[0]
    num_cols = 4    #4x2 in a batch
    num_rows = int(bs/num_cols)

    canvas = np.zeros((num_rows*224, num_cols*224*2, 3))
    img_idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            gt_img = gt_imgs[img_idx].permute(1,2,0).numpy()
            recon_img = recon_imgs[img_idx,:3,:,:].permute(1,2,0).numpy()
            canvas[i*224:(i+1)*224, j*224*2:(j+1)*224*2-224, :3] = gt_img
            canvas[i*224:(i+1)*224, j*224*2+224:(j+1)*224*2, :4] = recon_img
            img_idx += 1
    return (np.clip(canvas,0,1)*255).astype(np.uint8)


# ------------------------- train ---------------------------------------
def train(model, epoch):
    model.train()
    running_loss = []
    running_img_loss = []
    running_lmk_loss = []
    running_recog_loss = []
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, data in loop:
        in_img, gt_img, lmk = data
        in_img = in_img.to(device); lmk = lmk.to(device)
        gt_img = gt_img.to(device)
        optimizer.zero_grad()
        recon_params = model(in_img)
        loss,img_loss,lmk_loss,recog_loss,_ = face_loss(recon_params, gt_img, lmk)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        running_img_loss.append(img_loss.item())
        running_lmk_loss.append(lmk_loss.item())
        running_recog_loss.append(recog_loss.item())
        loop.set_description("Loss: {:.6f}".format(np.mean(running_loss)))

        if i % VERBOSE_STEP == 0 and i!=0:
            print ("Epoch: {:02}/{:02} Progress: {:05}/{:05} Loss: {:.6f} Img Loss: {:.6f} LMK Loss: {:.6f} Recog Loss {:.6f}".format(epoch+1,
                                                                                                                    NUM_EPOCH,
                                                                                                                    i,
                                                                                                                    len(train_dataloader),
                                                                                                                    np.mean(running_loss),
                                                                                                                    np.mean(running_img_loss),
                                                                                                                    np.mean(running_lmk_loss),
                                                                                                                    np.mean(running_recog_loss)))
            running_loss = []
            running_img_loss = []
            running_lmk_loss = []
            running_recog_loss = []

    return model

# ------------------------- eval ---------------------------------------
def eval(model, epoch):
    model.eval()
    all_loss_list = []
    img_loss_list = []
    lmk_loss_list = []
    recog_loss_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            in_img, gt_img, lmk = data
            in_img = in_img.to(device); lmk = lmk.to(device)
            gt_img = gt_img.to(device)

            recon_params = model(in_img)
            all_loss,img_loss,lmk_loss,recog_loss,recon_img=face_loss(recon_params, gt_img, lmk)
            all_loss_list.append(all_loss.item())
            img_loss_list.append(img_loss.item())
            lmk_loss_list.append(lmk_loss.item())
            recog_loss_list.append(recog_loss.item())
            if i == VISUAL_IDX:
                visualize_image = visualize_batch(gt_img, recon_img)

    print ("-"*50, " Test Results ", "-"*50)
    _all_loss = np.mean(all_loss_list)
    _img_loss = np.mean(img_loss_list)
    _lmk_loss = np.mean(lmk_loss_list)
    _recog_loss = np.mean(recog_loss_list)
    print ("Epoch {:02}/{:02} all_loss: {:.6f} image loss: {:.6f} landmark loss {:.6f} recog loss {:.6f}".format(epoch+1, NUM_EPOCH, _all_loss, _img_loss, _lmk_loss, _recog_loss))
    print ("-"*116)
    return _all_loss, _img_loss, _lmk_loss, _recog_loss, visualize_image


# Lists used for plotting loss

val_loss_list = []
img_loss_list = []
lmk_loss_list = []
#start training
for epoch in range(NUM_EPOCH):
    model = train(model, epoch) 
    all_loss, img_loss, lmk_loss, recog_loss, visualize_image = eval(model, epoch)
    val_loss_list.append(all_loss)
    img_loss_list.append(img_loss)
    lmk_loss_list.append(lmk_loss)
    lr_schduler.step(all_loss)
    io.imsave("./result/Epoch:{:02}_AllLoss:{:.6f}_ImgLoss:{:.6f}_LMKLoss:{:.6f}_RecogLoss:{:.6f}.png".format(epoch, all_loss, img_loss, lmk_loss, recog_loss), visualize_image)
    model2save = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(model2save, "./model_trained/epoch_{:02}_loss_{:.4f}_Img_loss_{:.4f}_LMK_loss{:.4f}_Recog_loss{:.4f}.pth".format(epoch+1, img_loss+lmk_loss, img_loss, lmk_loss, recog_loss))



# Plot training loss and validation loss

plt.plot(val_loss_list)
plt.plot(img_loss_list)
plt.plot(lmk_loss_list)
plt.title('trainging & validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['val_loss', 'img_pixel_loss','landmark_loss'], loc='upper left')
#save it
plt.savefig('train.png')
#plt.show()
















