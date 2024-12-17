Los códigos de preprocesamiento para datos de resonancia magnética 3D también se proporcionan paso a paso en el github. Si lo necesita, consulte [este enlace ](https://github.com/liqi814/Structural-Magnetic-Resonance-Imaging-sMRI-Pre-processing-Pipeline).

Este repositorio contiene los recibos para crear contenedores de singularity, o puede configurar el entorno usted mismo en función de los receipts. Consulte la carpeta [singularity_receipt](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/singularity_receipt).

## Citation
If you are using this repository, please cite this article
**Li Q, Yang MQ. 2021. Comparison of machine learning approaches for enhancing Alzheimer’s disease classification. PeerJ 9:e10549 https://doi.org/10.7717/peerj.10549**

## Singularity or Requirment

```bash
#buiding singularity containers
sudo singularity build resnet_cnn_mri.def classify.img
sudo singularity build cam_vis.img cam_vis.def
```
## 1. MRI Classification

### 1.1 Data Preparation (folder [data](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/data))

| File Name | Description |
| ------------- | ------------- |
| all_metadata.csv  | All image data |
| metadata.csv  | Training dataset  |
| test.csv  | Test dataset  |

### 1.2 Classification

```bash
module load singularity

##Use the VGG network:
nohup singularity exec --nv classify.img python3.5 scripts/vgg_3d_pred.py > vgg.out &

##Use the Resnet network:
nohup singularity exec --nv classify.img python3.5 scripts/res_3d_pred.py >resnet.out &
```

### 1.3 Result
 
Please check the folders [results_vgg](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/results_vgg) and [results_resnet](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/results_resnet) for more details.


## 2. Visualización Grad-CAM

### 1.1 Grad-CAM
Después de la clasificación, los modelos han aprendido los pesos a partir de imágenes. Se elige una imagen en la que le gustaría ver las regiones discriminativas. En este ejemplo, se utilizó S117504-reg.nii.gz. Y además, se puede cambiar a la capa que se quiera visualizar.


Uso:
```bash
##Without Singularity
python3.5  python_script  imgpath prefix
##With Singularity
Singularity  exec --nv classify.img  python3.5 python_script  imgpath  prefix
```

Por ejemplo:
```bash
qli@gpu001$ singularity exec --nv classify.img python3.5 scripts/vgg_3d_grad_cam.py /home/qli/AlzheimerClassify/5.Resize/S117504-reg.nii.gz S117504
Using gpu device 0: Tesla P100-PCIE-12GB (CNMeM is disabled, cuDNN 5110)
[[ 0.00110552  0.99889451]]
test_loss 0.001106127048842609
-----------
~/mri_classif


qli@gpu001$ singularity exec --nv classify.img python3.5 scripts/resnet_3d_grad_cam.py /home/qli/AlzheimerClassify/5.Resize/S117504-reg.nii.gz S117504
Using gpu device 0: Tesla P100-PCIE-12GB (CNMeM is disabled, cuDNN 5110)
[[ 0.12858798  0.87141204]]
test_loss 2.0511419773101807
-----------
~/mri_classif
```

A partir de los ejemplos, los scripts devolverán la siguiente información:
```bash
[[ 0.12858798  0.87141204]]
test_loss 2.0511419773101807
```
Los dos números dentro de los corchetes dobles representan las posibilidades de que esta imagen se clasifique como normal y Alzheimer respectivamente.

### 1.2 Depuración

Si se tiene un problema como el siguiente:
```bash
_tkinter.TclError: no display name and no $DISPLAY environment variable
```

Aquí está la solución:
```bash
echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
```


### 1.3 Visualización
Después de ejecutar Grad-CAM, se guardarán dos archivos .npz (uno que contiene la información discriminativa y otro que contiene los datos de MRI) en la carpeta [npz_res](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/npz_res) o [npz_vgg](https://github.com/liqi814/Deep-3D-CNNs-for-MRI-Classification-with-Alzheimer-s-Disease-And-Grad-CAM-Visualization/tree/master/npz_vgg). No es necesario ejecutar los scripts de esta sección en las tarjetas GPU.

```bash
#Usage:
singularity exec cam_vis.img python3.6 scripts/visualize_cam.py cam_npz/file/path mri_npz/file/path prefix

#Examples:

singularity exec cam_vis.img python3.6 scripts/visualize_cam.py npz_vgg/S117504_cam_conv4c.npz npz_vgg/S117504_mri.npz S117504_vgg

singularity exec cam_vis.img python3.6 scripts/visualize_cam.py npz_res/S117504_cam_voxres9_conv2.npz npz_res/S117504_mri.npz S117504_res
```
