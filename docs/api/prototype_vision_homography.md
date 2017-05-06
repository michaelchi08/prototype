# prototype.vision.homography



## Functions

- affine_transformation
- convert2homogeneous
- homography
- normalize

---


    affine_transformation(fp, tp)

Find Homography H, affine transformation, such that tp is affine transf
of fp.


---

    convert2homogeneous(points)

Convert a set of points (dim * n array) to homogeneous coordinates


---

    homography(fp, tp)

Find homography H, such that fp is mapped to tp using the linear DLT
method. Points are conditioned automatically


---

    normalize(points)

Normalize a collection of points in homogeneous coordinates so that the
last row equals 1


---
