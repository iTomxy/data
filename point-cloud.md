#  ModelNet40

- paper: [arXiv](https://arxiv.org/abs/1406.5670)
- project: [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://3dshapenets.cs.princeton.edu/)

CAD models, mesh, *.off* format, each representing one object, no noisy *background*,
for 3D object classification.

# ShapeNet

- project: [A Scalable Active Framework for Region Annotation in 3D Shape Collections](https://cs.stanford.edu/~ericyi/project_page/part_annotation/)

Objects shape, not background.

# S3DIS

Stanford 3D Indoor Scene dataset.
Indoor scene of 6 areas, including 271 rooms.
13 categories.
Has a `clutter` class.

Download:
- s3dis: https://cvg-data.inf.ethz.ch/s3dis/
- w/o XYZ: https://cvg-data.inf.ethz.ch/2d3ds/no_xyz/
- with XYZ: https://cvg-data.inf.ethz.ch/2d3ds/xyz/

## classes name

Check them in the [PointNet repository](https://github.com/charlesq34/pointnet):
- [pointnet/sem_seg/meta/class_names.txt](https://github.com/charlesq34/pointnet/blob/master/sem_seg/meta/class_names.txt)
```
ceiling
floor
wall
beam
column
window
door
table
chair
sofa
bookcase
board
clutter
```

Or check with [open3d](https://github.com/isl-org/Open3D) (on Linux):
```python
import open3d.ml.torch
print(open3d.ml.torch.datasets.S3DIS.get_label_to_names())
```
or
```python
import open3d.ml.tf
print(open3d.ml.tf.datasets.S3DIS.get_label_to_names())
```
Ref:
- [open3d.ml.tf.datasets.S3DIS](https://www.open3d.org/docs/latest/python_api/open3d.ml.tf.datasets.S3DIS.html#open3d.ml.tf.datasets.S3DIS)
- [open3d.ml.torch.datasets.S3DIS](https://www.open3d.org/docs/latest/python_api/open3d.ml.torch.datasets.S3DIS.html#open3d.ml.torch.datasets.S3DIS)

# SHREC15

Ojbect shape.
50 categories, each with 24 shapes. I.e. 1200 shapes in total.

# ScanNet

1513 indoor scenes.

# KITTI

Virtual KITTI 3D dataset.
Ourdoor scenes.
