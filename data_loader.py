import tensorflow_probability as tfp
import numpy as np

def shuffle_data(data: np.ndarray, labels: np.ndarray):
    """
    This function shuffles the data and labels while preserving the correspondence between them.
    
    Parameters:
        data (numpy.ndarray): The dataset to be shuffled.
        labels (numpy.ndarray): The labels corresponding to the dataset.
    
    Returns:
        tuple: Shuffled data and labels as two numpy arrays.
    """
    assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples"
    
    permutation = np.random.permutation(data.shape[0])
    
    shuffled_data = data[permutation]
    shuffled_labels = labels[permutation]
    
    return shuffled_data, shuffled_labels

def partition_data_single_client(iid, tumor_images, tumor_labels, num_classes, samples_per_client, alpha, available_indices):
    """
    This function partitions data for a single client.

    Parameters:
        tumor_images (numpy.ndarray): The dataset to be partitioned.
        tumor_labels (numpy.ndarray): The labels corresponding to the dataset.
        samples_per_client (int): The sample partitioned for each client from the original dataset.
        alpha (float): The alpha value from Dirichlet Distribution. 
        available_indices (numpy.ndarray): a range of indices for available data
    
    Returns:
        tuple: Partitioned clients data and their corresponding label arrays and used slices of data.
    """
    dirichlet_dist = tfp.distributions.Dirichlet(alpha)
    class_probs = dirichlet_dist.sample(1).numpy().reshape(-1) 

    class_counts = (class_probs * samples_per_client).astype(int)

    class_counts[-1] = max(1, class_counts[-1])

    total_count = np.sum(class_counts)
    if total_count != samples_per_client:
        diff = samples_per_client - total_count
        class_counts[np.argmax(class_counts)] += diff

    client_labels_partitioned = np.repeat(np.arange(1, num_classes + 1), class_counts)

    np.random.shuffle(client_labels_partitioned)

    client_data_partitioned = np.empty((samples_per_client, *tumor_images.shape[1:]), dtype=tumor_images.dtype)

    # mapping from each class to its indices in the dataset (adjust for 1-based labels)
    label_to_indices = {label: np.where(tumor_labels == label)[0] for label in range(1, num_classes + 1)}

    used_indices = set()
    for i, label in enumerate(client_labels_partitioned):
        label_indices = np.intersect1d(label_to_indices[label], available_indices)
        selected_index = np.random.choice(label_indices)
        client_data_partitioned[i] = tumor_images[selected_index]
        used_indices.add(selected_index)

    return client_data_partitioned, client_labels_partitioned, used_indices

def partition_data(iid:bool, num_clients:int, tumor_images: np.ndarray, tumor_labels: np.ndarray):

    """
    This function partitions data based on Dirichlet distribution using partition_data_single_client function

    Args:
        iid (bool): an attribute to define whether data should be partitioned in an iid or non-iid fashion
        if iid is set to True, alpha value is 10000.0 (iid data), if set to False, alpha value is 0.5 (non-iid data).
        num_clients (int): an attribute that create data partitions for n clients
        tumor_images (numpy.ndarray): The dataset to be partitioned.
        tumor_labels (numpy.ndarray): The labels corresponding to the dataset.

    Returns:
        a tuple containing two lists >>> client_data and client_labels
        Each list has N number of data partitions for N clients
    """
    tumor_images, tumor_labels = shuffle_data(tumor_images, tumor_labels)

    num_samples = len(tumor_images)
    num_classes = len(np.unique(tumor_labels))
    alpha = np.full(num_classes, 10000.0 if iid else 0.5) 

    samples_per_client = int(0.8 * num_samples / num_clients)

    client_data = []
    client_labels = []

    available_indices = np.arange(num_samples)

    for _ in range(num_clients):
        data, labels, used_indices = partition_data_single_client(iid, tumor_images, tumor_labels, num_classes, samples_per_client, alpha, available_indices)
        client_data.append(data)
        client_labels.append(labels)
        available_indices = np.setdiff1d(available_indices, np.array(list(used_indices)))

    return client_data, client_labels

def create_public_dataset(tumor_images: np.ndarray, tumor_labels: np.ndarray):
    """
    This function creates a public dataset for all clients from the shuffled tumor images and labels.
    
    Parameters:
        tumor_images (numpy.ndarray): The dataset of tumor images.
        tumor_labels (numpy.ndarray): The dataset of tumor labels.
    
    Returns:
        tuple: Public dataset of images and labels as two numpy arrays.
    """
    shuffled_images, shuffled_labels = shuffle_data(tumor_images, tumor_labels)
    
    # Use a percentage of the dataset for the public dataset, e.g., 10%
    public_dataset_size = int(0.5 * len(shuffled_images))
    
    public_images = shuffled_images[:public_dataset_size]
    public_labels = shuffled_labels[:public_dataset_size]
    
    return public_images, public_labels

def create_test_dataset(tumor_images: np.ndarray, tumor_labels: np.ndarray):
    """
    This function creates test datasets (x_test, y_test) from the shuffled tumor images and labels.
    
    Parameters:
        tumor_images (numpy.ndarray): The dataset of tumor images.
        tumor_labels (numpy.ndarray): The dataset of tumor labels.
    
    Returns:
        tuple: Test dataset of images and labels as two numpy arrays.
    """

    shuffled_images, shuffled_labels = shuffle_data(tumor_images, tumor_labels)
    
    test_dataset_size = int(0.5 * len(shuffled_images))
    
    x_test = shuffled_images[-test_dataset_size:]
    y_test = shuffled_labels[-test_dataset_size:]
    
    return x_test, y_test