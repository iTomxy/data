SYNTHIA [1] is a synthetic dataset of urban scenes,
created from a virtual city.
It has pixel-level annotations of `13` classes:
```
sky, building, road, sidewalk, fence,
vegetation, pole, car, sign, pedestrian,
cyclist, lane-marking, miscellaneous
```

It consists of several subsets
(*see the [Downloads](https://synthia-dataset.net/downloads/) page in [1]*):

- **SYNTHIA-Rand**.
Introduced in [1],
it contains `13,407` images of size 960x720.
Those images are not temporally consistent because this subset is not a video stream.
Images are annotated in class level,
but not instance level,
with `11` classes (and a `void`):
    ```
    void, sky, building, road, sidewalk,
    fence, vegetation, pole, car, sign,
    pedestrian, cyclist
    ```

- **SYNTHIA-Rand-Cityscapes**.
Introduced in [1],
it contains `9,400` images of size 720x1280.
Images are also not temporally consistent.
This subset provides instance level annotations of `23` classes:
    ```
    void, sky, building, road, sidewalk,
    fence, vegetation, pole, car, traffic sign,
    pedestrian, bicycle, motorcycle, parking-slot, road-work,
    traffic light, terrain, rider, truck, bus,
    train, wall, lanemarking
    ```
    This class set is compatible with the [Cityscapes](cityscapes.md) test set.

- **SYNTHIA-Seqs** / **SYNTHIA Video Sequences**.
Introduced in [1],
it contains `7` videos obtained from `8` views/cameras with frame size 960x720.
Each video is further divided into sub-sequences with same traffic situation
but under different weather/illumination/season condition.
This subset provides instance level pixel annotations of `13` classes:
    ```
    sky, building, road, sidewalk, fence,
    vegetation, pole, car, sign, pedestrian,
    cyclist, lane-marking, miscellaneous
    ```

- **SYNTHIA-SF**.
Introduced in [2],
it contains `6` videos,
each associating left & right image,
instance level pixel annotations and depth.
The class set is also Cityscapes compatiable:
    ```
    void, road, sidewalk, building, wall,
    fence, pole, traffic light, traffic sign, vegetation,
    terrain, sky, person, rider, car,
    truck, bus, train, motorcycle, bicycle,
    road lines, other, road works
    ```

- **SYNTHIA-AL**.
Introduced in [3],
it contains `1` video with annotations of instance level,
2D bounding boxes,
3D bounding boxes and depth.
The class set is:
    ```
    void, sky, building, road, sidewalk,
    fence, vegetation, pole, car, traffic sign,
    pedestrian, bycicle, lanemarking, traffic light
    ```

# References

1. (CVPR 2016) The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes - [paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.html), [project](https://synthia-dataset.net/), [paper with code](https://paperswithcode.com/dataset/synthia)
2. (BMVC 2017) Slanted Stixels: Representing San Francisco's Steepest Streets - [paper](https://bmva-archive.org.uk/bmvc/2017/papers/paper087/index.html), [blog](https://danihernandez.eu/slanted-stixels-representing-san-franciscos-steepest-streets-oral-bmvc2017/)
3. (ICCV Workshop 2019) Temporal Coherence for Active Learning in Videos - [paper](https://www.computer.org/csdl/proceedings-article/iccvw/2019/502300a914/1i5mkR5EAx2)
