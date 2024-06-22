import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
 
# Define the function to read the image
def read_img(img_path):
    '''
    Read image from img_path

    Parameters
    ----------
    img_path : str
        Path of image

    Returns
    -------
        Image (2D)
    '''
    try:
        img = plt.imread(img_path)
        return img
    except FileNotFoundError:
        print(f"No file found at {img_path}")
        return None

# Define the function to display the image
def show_img(img_2d):
    '''
    Show image

    Parameters
    ----------
    img_2d : <your type>
        Image (2D)
    '''
    plt.imshow(img_2d)    
    plt.show()
    
# Define the function to save the image
def save_img(img_2d, img_path):
    '''
    Save image to img_path

    Parameters
    ----------
    img_2d : <your type>
        Image (2D)
    img_path : str
        Path of image
    '''
    im = Image.fromarray(img_2d)
    im.save(img_path)
    
# Define the function to convert the 2D image to 1D
def convert_img_to_1d(img_2d):
    '''
    Convert 2D image to 1D image

    Parameters
    ----------
    img_2d : <your type>
        Image (2D)

    Returns
    -------
        Image (1D)
    '''
    height, width, channels = img_2d.shape
    img_1d = img_2d.reshape(width * height, channels)
    return img_1d

# explain the function

def update_centroids(x, labels, k):
  centroids = []
  for i in range(k):
    data_points = x[labels == i]

    # calculate the new centroid as the mean of the data points assigned to it
    if data_points.shape[0]:
        centroid = np.mean(data_points, axis=0)
        print(centroid)
        centroids.append(centroid)
  return np.array(centroids)

def assign_labels(x, centroids):
#   labels = []
#   for i in range(len(x)):
#     # calculate the distance between the data point and each centroid
#     distances = np.linalg.norm(centroids - x[i], axis=1)
#     # assign the data point to the nearest centroid
#     labels.append(np.argmin(distances))
#   return labels

    # calculate the distance between each data point and each centroid
    return np.argmin(np.linalg.norm(x - centroids[:, None], axis=2), axis=0)

# explain the function


def kmeans(x, k, no_of_iterations, init_centroids='random'):
    '''
    K-Means algorithm

    Parameters
    ----------
    img_1d : np.ndarray with shape=(height * width, num_channels)
        Original (1D) image
    k_clusters : int
        Number of clusters
    max_iter : int
        Max iterator
    init_centroids : str, default='random'
        The method used to initialize the centroids for K-means clustering
        'random' --> Centroids are initialized with random values between 0 and 255 for each channel
        'in_pixels' --> A random pixel from the origiknal image is selected as a centroid for each cluster

    Returns
    -------
    centroids : np.ndarray with shape=(k_clusters, num_channels)
        Stores the color centroids for each cluster
    labels : np.ndarray with shape=(height * width, )
        Stores the cluster label for each pixel in the image
    '''
    if init_centroids == 'random':
        # Centroids are initialized with random values between 0 and 255 for each channel
        centroids = np.random.randint(0, 256, size=(k, x.shape[1]))
    elif init_centroids == 'in_pixels':
        # choose random pixels as centroids
        centroids = x[np.random.choice(range(len(x)), k, replace=False)]
    else:
        raise ValueError("Invalid initialization method")

    for _ in range(no_of_iterations):
        # assign each data point to the nearest centroid
        labels = assign_labels(x, centroids)

        old_centroids = centroids

        # update centroids based on the assigned data points
        centroids = update_centroids(x, labels, k)

        if np.array_equal(old_centroids, centroids):
            break

    return labels, centroids


# Generate a 2D image based on K-means cluster centroids 
def generate_2d_img(img_1d, img_2d_shape, centroids, labels):
    '''
    Generate a 2D image based on K-means cluster centroids

    Parameters
    ----------
    img_2d_shape : tuple (height, width, 3)
        Shape of image
    centroids : np.ndarray with shape=(k_clusters, num_channels)
        Store color centroids
    labels : np.ndarray with shape=(height * width, )
        Store label for pixels (cluster's index on which the pixel belongs)

    Returns
    -------
        New image (2D)
    '''
    # Create an empty image array using np.zeros_like()
    img = np.zeros_like(img_1d)

    # Assign the pixel values to the centroids
    for i in range(len(img)):
        img[i] = centroids[labels[i]]
    # Reshape the image to 2D
    img = img.reshape(img_2d_shape)
    return img

# How to implement elbow method to find the optimal number of clusters
def elbow_method(x, max_clusters):
    '''
    Implement the elbow method to find the optimal number of clusters

    Parameters
    ----------
    x : np.ndarray with shape=(height * width, num_channels)
        Original (1D) image
    max_clusters : int
        Maximum number of clusters to consider

    Returns
    -------
    distortions : list
        List of distortion values for each number of clusters
    '''
    distortions = []
    for k in range(1, max_clusters + 1):
        labels, centroids = kmeans(x, k, no_of_iterations=10)
        distortion = np.mean(np.linalg.norm(x - centroids[labels], axis=1))
        distortions.append(distortion)
    return distortions
    
def main():
  compress_file = input("Enter the path of the image file: ")
  k_clusters = int(input("Enter the number of clusters: "))
  max_iter = int(input("Enter the maximum number of iterations: "))
  init_centroids = input("Enter the method to initialize the centroids (random/in_pixels): ")
  image = read_img(compress_file)

  img_1d = convert_img_to_1d(image)
  labels, centroids = kmeans(img_1d, k_clusters, max_iter, init_centroids)
  new_image = generate_2d_img(img_1d, image.shape, centroids, labels)

  print("Processing complete!")

  show_option = input("Do you want to display the compressed image? (y/n): ")
  
  if show_option == 'y':
    show_img(new_image)  

  save_option = input("Do you want to save the compressed image? (y/n): ")

  if save_option == 'y':
    save_at_current_path = input("Do you want to save the image at the current path? (y/n): ")
    
    file_name = input("Enter the file name: ")
    file_type = input("Enter the file type (png/pdf): ")

    file = f"{file_name}.{file_type}"

    if save_at_current_path == 'y':
        save_img(new_image, f"{file}")
        print("Image saved at the current path!")
    else:
        base_path = input("Enter the base path: ")

        # Process the path
        path = base_path + '\\' + file
        save_img(new_image, path)

        print(f"Image saved at {path}")
    # image = read_img("mountain.jpg")
    # img_1d = convert_img_to_1d(image)
    # distortions = elbow_method(img_1d, max_clusters=10)
    # plt.plot(range(1, 11), distortions)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Distortion')
    # plt.title('Elbow Method')
    # plt.show()

if __name__ == "__main__":
    main()  # Call the main function






