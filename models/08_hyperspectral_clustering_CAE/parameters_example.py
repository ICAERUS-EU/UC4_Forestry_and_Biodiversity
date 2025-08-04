params = {
            "num_clusters": 5,  # The cluster number to find
            "inshape": [9, 9, 224],  # shape of the part of hyperspectral image, sugested square shape with all spectral bands (224 with our Specim data). 
            "kernel": [3, 3, 19],  # kernel must be smaller than the inshape. 
            "n_epochs": 10,  # training epochs
            "update_interval": 100,  # number of training batches between the clustering distribution recalculation (used for optimization since it takes a lot of time to calculate, possible rewrite of kmeans to GPU might help).
            "gamma": 0.001,  # clustering loss multiplier.
            "batch_size": 1024 * 2,  # nuber of inshape parts to use at the same time for training
            "LR": 0.0008,   # optimizer learning rate
            "weight_decay": 0.0001,  # optimizer weight decay
            "latent_space": 25,  # clustering AE latent space size
            "filters": 6,  # clustering AE convolution filter size (keep around the number of clusters, but depends on the datasets used)
            "smoothing": True  # to use SAVGOL spectral smoothing for training and predictions or not.
         }