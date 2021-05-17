# 3DFaceReconstruction
The implementation of the 3D face reconstruction through Resnet and BFM model, for the course project.
## Preparing environment
use Singularity on linux(with GPU)
## Preparing Dataset
Download and unzip the CACD original face from 
>Original face images (detected and croped by openCV face detector) can be downloaded [here](https://drive.google.com/file/d/1hYIZadxcPG27Fo7mQln0Ey7uqw1DoBvM/view?usp=sharing) (3.5G)
>
Modify the corrsponding path of the dataset input `image_list = glob.glob("../carc/CACD2000/*.jpg")` and dataset out put
`save_path=./data/CACD2000_{}.hdf5`in the `Dataprepocessing.py`. Modify the `SHORT_LEN` to get a smaller subdataset.

Run `python Dataprepocessing.py`

## Training the model
Check the path in and run `python Trainnet.py`
## Testing the model and get outcome
Check the path in and run `python Trainnet.py`
