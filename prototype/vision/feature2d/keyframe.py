class KeyFrame:
    """KeyFrame

    Parameters
    ----------
    image : np.array
        Image

    features : list of Features
        Features

    """

    def __init__(self, image, features):
        self.image = image
        self.features = features

    def update(self, image, features):
        """Update

        Parameters
        ----------
        image : np.array
            Image

        features : list of Feature
            Features

        """
        self.image = image
        self.features = features
