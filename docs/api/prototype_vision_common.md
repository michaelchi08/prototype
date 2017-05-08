# prototype.vision.common



## Functions

- camera_center
- camera_intrinsics
- factor_projection_matrix
- focal_length
- projection_matrix
- random_3d_features

---


    camera_center(P)

Extract camera center from projection matrix P 

---

    camera_intrinsics(fx, fy, cx, cy)

Construct camera intrinsics matrix K 

---

    factor_projection_matrix(P)

Extract camera intrinsics, rotation matrix and translation vector 

---

    focal_length(image_width, image_height, fov)

Calculate focal length in the x and y axis from:
- image width
- image height
- field of view


---

    projection_matrix(K, R, t)

Construct projection matrix from:
- Camera intrinsics matrix K
- Camera rotation matrix R
- Camera translation vector t


---

    random_3d_features(nb_features, feature_bounds)

Generate random 3D features 

---
