import cv2
import os
import sys
import pickle
import random 

import numpy as np

from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import KDTree

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model


# ----------------------------------------------
#  EXCEPTION CLASSES
# ----------------------------------------------

class ImageSizeIncompatible(Exception):
    '''
    Exception raised when an image size is imcompatible.

    Attributes:
        message -- error message
    '''

    def __init__(self, message: str = "Image size incompatible"):
        self.message = message
        super().__init__(self.message)


class DatabaseAlreadyExists(Exception):
    '''
    Exception raised when a database initialized using the name
    and location of one that already exists.

    Attributes:
        message -- error message
    '''

    def __init__(self, message: str = "Database already exists"):
        self.message = message
        super().__init__(self.message)


class DatabaseNotFound(Exception):
    '''
    Exception raised when a database is not found.

    Attributes:
        message -- error message
    '''

    def __init__(self, message: str = "Database not found"):
        self.message = message
        super().__init__(self.message)


# ----------------------------------------------
#  FEATURE EXTRACTOR FUNCTIONS
# ----------------------------------------------

def getFeatureExtractor() -> Model:
    '''
    Returns a Keras Model.

    Feature extractor is generated from a VGG16 model pre-trained on ImageNet
    by removing the final classification layer.

    Model Input Size: 224x224x3
    Model Output Size: 1x4096
    '''
    vgg16 = VGG16(include_top=True, 
                  weights="imagenet", 
                  input_shape=None, 
                  pooling=None, 
                  classes=1000) 
    dense_layer = vgg16.get_layer(name="fc1")
    feature_extractor = Model(inputs=vgg16.input, outputs=dense_layer.output)
    
    return feature_extractor


def getFeatures(input_image: np.array) -> np.array:
    '''
    Uses "getFeatureExtractor" to generate a model which input_image is fed through.

    :param input_image:
        Numpy array that will be used as the input to the pre-trained VGG16 model.
    '''
    feature_extractor = getFeatureExtractor()

    if len(input_image.shape) != 3:
        raise ImageSizeIncompatible()

    slice_arr = cv2.resize(input_image.astype(np.float32), (224, 224))
    slice_arr = preprocess_input(slice_arr)

    slice_image_features = feature_extractor.predict(np.expand_dims(slice_arr, axis=0))

    return slice_image_features


# ----------------------------------------------
#  IMAGE PROCESSING/DATABASE FUNCTIONS
# ----------------------------------------------

def initDB(db_loc: str, name: str) -> bool:
    '''
    Initializes the base structure of a database

    :param db_loc:
        Directory to initialize database.

    :param name:
        Desired name of database.

    File Struture:
    |-- ImageDatabase
    │   |-- ImportedImages
    |   |-- ProcessedImages
    |   |   |-- <size>x<size>
    |   |   |   |-- Images
    |   |   |   |-- Embedded
    |   |   |   |-- kdtree.p
    '''
    db_loc = Path(db_loc)
    db_dir = db_loc.joinpath(name)
    if db_dir.is_dir():
        raise DatabaseAlreadyExists("Unable to create database at location {}: already exists".format(db_dir))
    else:
        imported_dir  = db_dir.joinpath("ImportedImages")
        processed_dir = db_dir.joinpath("ProcessedImages")
        
        os.mkdir(db_dir)
        os.mkdir(imported_dir)
        os.mkdir(processed_dir)
        return True


def getRandomIds(database_loc: str, random_count: int) -> list:
    '''
    Returns a list of random image ids (used to download image from flask server) and autoincremented index id.

    :param database_loc:
        Location to database.

    :param random_count:
        Number of random ids to generate
    '''
    image_dir = Path(database_loc).joinpath("ProcessedImages").joinpath("256x256").joinpath("Images")

    if not image_dir.is_dir():
        raise DatabaseNotFound(f"No database found at location {database_loc}")

    random_samples = random.sample([p for p in image_dir.iterdir()], random_count)

    selected_ids_and_idxs = []

    for samp in random_samples:
        _id = str(samp).replace("/", "_")
        idx = samp.name.split("-")[0]
        selected_ids_and_idxs.append({"id": _id, "idx": int(idx)})

    return selected_ids_and_idxs


def sliceImage(img_arr: np.array, new_size: tuple) -> list:
    '''
    Creates slices of an image.

    :param img_arr:
        image array used for slicing

    :param new_size: 
        width and height of new slices
    
    Returns: 
        List of tuples containing the starting pixel location of the slice and the sliced image. 
        If slicing was unable to be completed then an emtpy list is returned.
        i.e. [ (x_start_pix, y_start_pix, sliced_image_array) ]
    '''
    width      = img_arr.shape[0]
    height     = img_arr.shape[1]
    new_width  = new_size[0]
    new_height = new_size[1]

    # Return empty dictionary if desired slice is larger than image
    if new_width > width or new_height > height:
        return {}

    # Return data
    image_list = []

    # Process image
    for i in range(0, width, new_width):
        for j in range(0, height, new_height):
            
            # Create array of zeros of slice size
            base = np.zeros((new_width, new_height, img_arr.shape[2]))

            # Slice image and broadcase to base array
            # Slice will be blacked out when outside range of original image
            slice_arr = img_arr[i:i+new_width, j:j+new_height, :]
            base[ :slice_arr.shape[0], :slice_arr.shape[1]] = slice_arr
            image_list.append((i, j, base))

    return image_list


def importAndProcess(img_path: str, db_path: str, new_img_name: str, slice_sizes: list = ["256"], pixels_not_blank: float = 0.85) -> None:
    '''
    Main image ingestion function.

    :param img_path:
        Path of image to import into database.

    :param db_path:
        Location of database. Must be a DB initialized using function "initDB."

    :param new_img_name:
        Name to use as the base name in storage.

    :param slice_sizes:
        List of slice sizes to generate.

    :param pixels_not_blank:
        Percentage of images pixels required to be not blank for image to be saved.
    '''
    img_path = Path(img_path)
    db_path  = Path(db_path)

    # Remove kdtree and embedded projections   
    if db_path.joinpath("ProcessedImages/256x256/kdtree.p").is_file():
        os.remove(db_path.joinpath("ProcessedImages/256x256/kdtree.p"))

    # Ensure database exists
    if not db_path.is_dir():
        raise DatabaseNotFound(f"No database found at location {db_path}")

    # Internal database directory
    images_path = db_path.joinpath("ImportedImages")

    # Ensure database was initialized correctly
    if not images_path.is_dir():
        raise DatabaseNotFound(f"Database at location {db_path} not properly initialized. Use function 'initDB'")

    # PROCESS IMAGE

    # Step 1: Read in image
    img_arr  = cv2.imread(str(img_path))
    file_ext = "png"

    # Auto increment id
    db_ids = [int(f.name.split("-")[0]) for f in images_path.iterdir() ]
    db_id = 0  
    if len(db_ids) != 0: db_id = max(db_ids) + 1 

    # Step 2: Save Original File
    saved_image = images_path.joinpath(f"{db_id}-{new_img_name}.{file_ext}")
    cv2.imwrite(str(saved_image), img_arr)

    processed_image_dir = db_path.joinpath("ProcessedImages")  

    for size in slice_sizes:
        print("Processing size:", size)

        # Run processing
        slice_image_list = sliceImage(img_arr, (int(size), int(size)))

        feature_extractor_model = getFeatureExtractor()
        feature_extractor_model.summary()

        slice_size_dir = processed_image_dir.joinpath(f"{size}x{size}")
        slice_size_image_dir    = slice_size_dir.joinpath("Images")
        slice_size_embedded_dir = slice_size_dir.joinpath("Embedded")
        
        if not slice_size_dir.is_dir():
            os.mkdir(slice_size_dir)
            os.mkdir(slice_size_image_dir)
            os.mkdir(slice_size_embedded_dir)

        # Generate slice id
        slice_ids = [int(f.name.split("-")[0]) for f in slice_size_image_dir.iterdir() ]
        slice_id = 0  
        if len(slice_ids) != 0: slice_id = max(slice_ids) + 1 

        # Save all sliced images
        for i in tqdm(range(len(slice_image_list))):
            x = slice_image_list[i][0]
            y = slice_image_list[i][1]
            slice_arr = slice_image_list[i][2]

            nonzero_pixel_count = np.count_nonzero(slice_arr)
            total_pixel_count = slice_arr.shape[0] * slice_arr.shape[1] * slice_arr.shape[2]

            if (nonzero_pixel_count / total_pixel_count) >= pixels_not_blank:
                slice_saved_image = slice_size_image_dir.joinpath("{id}-{parent_id}-{image_name}-{pix_x}-{pix_y}.{ext}"
                                                                  .format(id=slice_id, 
                                                                          parent_id=db_id, 
                                                                          image_name=new_img_name, 
                                                                          pix_x=x, 
                                                                          pix_y=y, 
                                                                          ext=file_ext))
                slice_arr = ((slice_arr / slice_arr.max())*255).astype(np.int)
                cv2.imwrite(str(slice_saved_image), slice_arr)

                # Extract and save features
                slice_arr = cv2.resize(slice_arr.astype(np.float32), (224, 224))
                slice_arr = preprocess_input(slice_arr)

                slice_image_features = feature_extractor_model.predict(np.expand_dims(slice_arr, axis=0))
                features_image_path = slice_size_embedded_dir.joinpath("{id}-{parent_id}-{image_name}-{pix_x}-{pix_y}.npy"
                                                                       .format(id=slice_id, 
                                                                               parent_id=db_id, 
                                                                               image_name=new_img_name, 
                                                                               pix_x=x, 
                                                                               pix_y=y))

                with open(features_image_path, "wb") as f:
                    np.save(f, slice_image_features[0])
                
                slice_id += 1


# ----------------------------------------------
#  QUERY FUNCTIONS
# ----------------------------------------------

def getKDTreeAndFeatureVectors(embedded_dir: str) -> (KDTree, np.array):
    '''
    Returns the current KDTree or generates one if there isn't one saved,
    as well as a list of embedded feature vectors, in order by their 
    autoincremented integer id.

    :params embedded_dir:
        Location of saved embedded feature vectors.
    '''

    EMBEDDED_VECTOR_SIZE = 4096

    embedded_dir = Path(embedded_dir)
    tree_file_path = embedded_dir.parent.joinpath("kdtree.p")

    embedded_np_files = [p for p in embedded_dir.iterdir()]
    embedded_np_files = sorted(embedded_np_files, key=lambda p: int(p.name.split("-")[0]))
    embedded_vectors = np.zeros((len(embedded_np_files), EMBEDDED_VECTOR_SIZE))

    for i in range(len(embedded_np_files)):
        embedded_vectors[i] = np.load(embedded_np_files[i])

    tree = None

    # Generate tree and save if there isnt one saved
    if not tree_file_path.is_file():
        tree = KDTree(embedded_vectors, metric="manhattan")
        pickle.dump(tree, open(str(tree_file_path), "wb"))
        
    # Load and return a saved KDTree
    else:
        tree = pickle.load(open(str(tree_file_path), "rb"))

    return tree, embedded_vectors


def queryImageIdx(image_idx: int, database_loc: str, query_count: int) -> list:
    '''
    Query the database for similar images. Returns in order of relevance. 
    NOTE: The current implementation will return the queried queried image as part
          of the result. Distance will be equal to 0. Therefore, the resulting list
          will be one larger than "query_count"

    :param image_idx:
        The unique autoincremented integer ID associated with an image in the database.

    :param database_loc:
        Location of database.

    :param query_count:
        Desired number of images in query results

    '''

    embedded_dir = f"{database_loc}/ProcessedImages/256x256/Embedded"

    if not Path(embedded_dir).is_dir():
        raise DatabaseNotFound(f"No database found at location {database_loc}")

    tree, embedded_vectors = getKDTreeAndFeatureVectors(embedded_dir)

    # Sort images paths on autoincremented value so that 'image_idx' corresponds to an
    # index into the 'image_paths' array
    image_dir = Path(embedded_dir).parent.joinpath("Images")
    image_paths = [p for p in image_dir.iterdir()]
    image_paths = sorted(image_paths, key=lambda p: int(p.name.split("-")[0]))

    # Query Tree
    dist, ind = tree.query(np.expand_dims(embedded_vectors[image_idx], axis=0), k=query_count+1)

    dist = dist[0]
    ind  = ind[0]

    query_images = []

    for i in range(len(ind)):
        image_id = str(image_paths[ind[i]]).replace("/", "_")
        query_images.append( dict(id=image_id, idx=int(ind[i]), distance=int(dist[i])) )

    return query_images