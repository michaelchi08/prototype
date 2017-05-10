# prototype.vision.utils



## Functions

- camera_intrinsics
- focal_length
- generate_random_3d_features
- projection_matrix

---


    camera_intrinsics(fx, fy, cx, cy)

Construct camera intrinsics matrix `K` from focal length `fx` and `fy`
    and camera principle center `cx` and `cy` 


---

    focal_length(image_width, image_height, fov)

Calculate focal length `(fx, fy)` from `image_width`, `image_height` and
    field of view `fov` 


---

    generate_random_3d_features(bounds, nb_features)

Generate random 3d features

Args:

        bounds (dict): 3D feature bounds, for example
                       bounds = {
                         "x": {"min": -1.0, "max": 1.0},
                         "y": {"min": -1.0, "max": 1.0},
                         "z": {"min": -1.0, "max": 1.0}
                       }

        nb_features (int): number of 3D features to generate

    Returns:

        features (list): list of 3D features in homogeneous coordinate

    


---

    projection_matrix(K, R, t)

Construct projection matrix from camera intrinsics `K`, camera rotation
    `R` and camera translation `t` 


---
