Cityscapes [1] is an urban scene image dataset
containing `25,000` images,
5,000 of which are with high quality pixel annotations of `30` classes,
while the other 20,000 are coarsely annotated.
The class set is:
```
(flat): road, sidewalk, parking+, rail, track+
(human): person*, rider*
(vehicle): car*, truck*, bus*, on rails*, motorcycle*, bicycle*, caravan*+, trailer*+
(construction): building, wall, fence, guard rail+, bridge+, tunnel+
(object): pole, pole group+, traffic sign, traffic light
(nature):, vegetation, terrain
(sky): sky
(void): ground+, dynamic+, static+
```
where:
- `*` means instance annotations are available for this class;
- `+` means this clss is not included in any evaluation and treated as void.

# References

1. (CVPR 2016) The Cityscapes Dataset for Semantic Urban Scene Understanding - [paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html), [project](https://www.cityscapes-dataset.com/), [paper with code](https://paperswithcode.com/dataset/cityscapes)
