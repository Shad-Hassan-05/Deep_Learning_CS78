import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets


def create_dataset(data_path, output_path=None, contrast_normalization=False, whiten=False, transforms=False):
    """
    Reads and optionally preprocesses the data.

    Arguments
    --------
    data_path: (String), the path to the file containing the data
    output_path: (String), the name of the file to save the preprocessed data to (optional)
    contrast_normalization: (boolean), flags whether or not to normalize the data (optional). Default (False)
    whiten: (boolean), flags whether or not to whiten the data (optional). Default (False)

    Returns
    ------
    train_ds: (TensorDataset, the examples (inputs and labels) in the training set
    val_ds: (TensorDataset), the examples (inputs and labels) in the validation set
    """
    # read the data and extract the various sets
    data = torch.load(data_path, weights_only = False)

    # extract data
    data_tr = data['data_tr']
    sets_tr = data['sets_tr']
    label_tr = data['label_tr']
    data_te = data['data_te']
    class_names = data['class_names']

    # apply the necessary preprocessing as described in the assignment handout.
    # You must zero-center both the training and test data
    if data_path == "image_categorization_dataset.pt":
        
        # use only training images, so create a mask using sets_tr values
        # all trying data should have value 1 in sets_tr
        training_mask = (sets_tr == 1)

        # calculate per-pixel mean using only training data
        num_images = data_tr[training_mask].shape[0] # get the number of images total
        pr_pix_mean = data_tr[training_mask].mean(dim = 0) # calculate the per pixel mean (3x32x32)

        # zero centered data; testing and training.
        data_tr = data_tr - pr_pix_mean # zero-center the training data (Nx3x32x32)
        data_te = data_te - pr_pix_mean  # zero-center the test data (Nx3x32x32)
            
        # %%% DO NOT EDIT BELOW %%%% #
        if contrast_normalization:
            image_std = torch.std(data_tr[sets_tr == 1], unbiased=True)
            image_std[image_std == 0] = 1
            data_tr = data_tr / image_std
            data_te = data_te / image_std
        if whiten:
            examples, rows, cols, channels = data_tr.size()
            data_tr = data_tr.reshape(examples, -1)
            W = torch.matmul(data_tr[sets_tr == 1].T, data_tr[sets_tr == 1]) / examples
            E, V = torch.linalg.eigh(W)
            E = E.real
            V = V.real

            en = torch.sqrt(torch.mean(E).squeeze())
            M = torch.diag(en / torch.max(torch.sqrt(E.squeeze()), torch.tensor([10.0])))

            data_tr = torch.matmul(data_tr.mm(V.T), M.mm(V))
            data_tr = data_tr.reshape(examples, rows, cols, channels)

            data_te = data_te.reshape(-1, rows * cols * channels)
            data_te = torch.matmul(data_te.mm(V.T), M.mm(V))
            data_te = data_te.reshape(-1, rows, cols, channels)

        preprocessed_data = {"data_tr": data_tr, "data_te": data_te, "sets_tr": sets_tr, "label_tr": label_tr}
        if output_path:
            torch.save(preprocessed_data, output_path)

    train_ds = TensorDataset(data_tr[sets_tr == 1], label_tr[sets_tr == 1])
    val_ds = TensorDataset(data_tr[sets_tr == 2], label_tr[sets_tr == 2])

    return train_ds, val_ds

