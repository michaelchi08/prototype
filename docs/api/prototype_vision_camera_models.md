# prototype.vision.camera_models

## Classes

- PinHoleCameraModel


### PinHoleCameraModel


    __init__(self, image_width, image_height, hz, K)


---

    check_features(self, dt, features, rpy, t)

Check whether features are observable by camera 

---

    project(self, X, R, t)

Project 3D point to image plane 

---

    update(self, dt)

Update camera

---

