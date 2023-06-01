# tract-segmentation
## Get Started


## Install
* PyTorch >= 3.6
* [Mrtrix 3](https://mrtrix.readthedocs.io/en/latest/installation/build_from_source.html) >= 3.0
``conda install -c mrtrix3 mrtrix3 python=3.6``

* boto3
``pip install boto3``

* tractseg
``pip install tractseg``

* nibabel
``pip install nibabel``
## Datasets Prepare
You can prepare datasets by yourself or follow the following steps.
* Download Human Connectome Project (HCP) datasets.
1. Register a HCP account: [HCP](https://db.humanconnectome.org/)
2. Enable Amazon S3 Access: [AWS](https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS)
3. Download HCP datasets by running [download_HCP_1200_diffusion_mri.py](/download_HCP_1200_dMRI.py):

``python /download_HCP_1200_diffusion_mri.py --id your_aws_id --key your_aws_key --out_dit your_hcp_dir``
* Download Corresponding WM tract labels from [Zenodo](https://zenodo.org/record/1477956#.ZBQ5wHZByNc).
## Data Pre-Processing
* Transform a trk streamline file to a binary map.
Transform a trk streamline file to a binary map by running [trk2bin.py](/trk2bin.py):

``python /trk2bin.py --tract_dir your_tract_dir --ref_dir your_hcp_dir``

and finally, the tract dataset directory should look like:

    $your_tract_dir
    ├─992774
    │   ├─tracts
    │   │   ├─AF_left.nii.gz
    │   │   ├─AF_rgiht.nii.gz
    |   |   .
    |   |   .
    |   |   .
    │   │   ├─UF_rgiht.nii.gz
    ├─991267
    │   ├─tracts
    │   │   ├─AF_left.nii.gz
    │   │   ├─AF_rgiht.nii.gz
    |   |   .
    |   |   .
    |   |   .
    │   │   ├─UF_rgiht.nii.gz
    .
    .
    .
    ├─599469
    │   ├─tracts
    │   │   ├─AF_left.nii.gz
    │   │   ├─AF_rgiht.nii.gz
    |   |   .
    |   |   .
    |   |   .
    │   │   ├─UF_rgiht.nii.gz
  

* Transform dMRI datasets to peak data.
Transform dMRI datasets to peak data using multi-shell multi-tissue constrained spherical deconvolution (MSMT-CSD) by running [HCP2MSMT_CSD.py](/HCP2MSMT_CSD.py):

``python /HCP2MSMT_CSD.py --hcp_dir your_hcp_dir --out_dir your_msmt_csd_dir``

and finally, the peak data directory should look like:

    $your_msmt_csd_dir
    ├─992774
    │   ├─peaks.nii.gz
    ├─991267
    │   ├─peaks.nii.gz
    .
    .
    .
    ├─599469
    │   ├─peaks.nii.gz
## Training
### 1. Train base tract segmentation model (Teacher model)
``python \train_model.py --action train_base_tract --data_dir your_data_dir --label_dir your_label_dir --ratio 1/2/5 --ckpt_dir your_ckpt_dir``
### 2. Train novel tract segmentation model (Student model)
``python \train_model.py --action train_novel_tract --data_dir your_data_dir --label_dir your_label_dir --ratio 1/2/5 --ckpt_dir your_ckpt_dir``
## Testing
``python \train_model.py --action test_novel_tract --data_dir your_data_dir --label_dir your_label_dir --ratio 1/2/5 --ckpt_dir your_ckpt_dir``

## Comparison
### CFT
#### Train
``python \train_model.py --action train_CFT --data_dir your_data_dir --label_dir your_label_dir --ratio 1/2/5 --ckpt_dir your_ckpt_dir``
#### Test
``python \train_model.py --action test_CFT --data_dir your_data_dir --label_dir your_label_dir --ratio 1/2/5 --ckpt_dir your_ckpt_dir``

### IFT
#### Train
``python \train_model.py --action train_IFT --data_dir your_data_dir --label_dir your_label_dir --ratio 1/2/5 --ckpt_dir your_ckpt_dir``
#### Test
``python \train_model.py --action test_IFT --data_dir your_data_dir --label_dir your_label_dir --ratio 1/2/5 --ckpt_dir your_ckpt_dir``


### TractSeg
#### Train
``python \train_model.py --action train_TractSeg --data_dir your_data_dir --label_dir your_label_dir --ratio 1/2/5 --ckpt_dir your_ckpt_dir``
#### Test
``python \train_model.py --action test_TractSeg --data_dir your_data_dir --label_dir your_label_dir --ratio 1/2/5 --ckpt_dir your_ckpt_dir``

## Ackonlwdgement
Our code is based on [TractSeg](https://github.com/MIC-DKFZ/TractSeg).
