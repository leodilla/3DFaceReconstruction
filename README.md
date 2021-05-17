# 3DFaceReconstruction
The implementation of the 3D face reconstruction through Resnet and BFM model, for the course project.
## Preparing environment
use Singularity on linux(with GPU)
/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif
Need python=3.7.0, pytorch=1.1.0, torchvision=0.3.0, cudatoolkit=10.0
In singularity container, you may need
```
Singularity>git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch

Singularity>pip install h5py
```
## Preparing Dataset
Download and unzip the CACD original face from 
>Original face images (detected and croped by openCV face detector) can be downloaded [here](https://drive.google.com/file/d/1hYIZadxcPG27Fo7mQln0Ey7uqw1DoBvM/view?usp=sharing) (3.5G)
>
Modify the corrsponding path of the dataset input `image_list = glob.glob("../carc/CACD2000/*.jpg")` and dataset out put
`save_path=./data/CACD2000_{}.hdf5`in the `Dataprepocessing.py`. Modify the `SHORT_LEN` to get a smaller subdataset.

Run `python Dataprepocessing.py`

## Training the model
Check the path in `dataloader=("./data/CACD2000_train.hdf5")`and you will get the trained model in `./model_trained/`, with the visualized output images in `./result`, and the loss curve in `train.png`.

Default `NUM_EPOCH=25, BATCH_SIZE=8`

Run `python trainnet.py`
## Testing the model and get outcome
Check the path in `dataloader=("./data/CACD2000_test.hdf5")`and you need to put the model into `MODEL_LOAD_PATH ="./modelload/demo25.pth"`, and you will get all of the visualized output images in `./test`

Run `python testnet.py`

