import pandas as pd
from tensorflow.keras.applications import DenseNet169, ResNet50, VGG16
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.metrics import CohenKappa

from image_preprocessing import rescale

DATASET_PATH = '/content/MURA-v1.1/tvt_detailed_paths.csv'
ROOT_DIR = '/content'


def get_dataframe(body_part, split, path=DATASET_PATH):
    """
    Filters dataset by body part and data split

    Parameters
    ----------
    body_part : str
        Name of specific body part / all body parts (no filtering)
    split : str
        Name of specific data split / all images (no filtering)
    path : str or None
        Path to dataset file, if left empty, default path is used

    Returns
    -------
    pd.Dataframe
        Dataframe containing filtered dataset
    """
    if body_part == 'ALL':
        body_part = '.*'
    if split == 'ALL':
        split = '.*'

    df = pd.read_csv(path)
    filtered_df = df.loc[(df.body_part.str.match(body_part, case=False)) & (df.split.str.match(split, case=False))]
    return filtered_df


def build_model(base_name, weights, shape, name, pooling='max', optimizer=Adam(learning_rate=0.0001),
                metrics=[CohenKappa(num_classes=2, name='kappa')], add_top=False):
    """
    Builds and compiles a CNN model using one of tf.keras.applications pre-built architectures.

    Parameters
    ----------
    base_name : str
        Name of the desired pre-built tf.keras.applications architecture
    weights : str or None
        What weights should the model be initialized with. None for random, "imagenet" for using weights
        pre-trained on ImageNet dataset or a path to load weights from.
    shape : tuple(int, int, int)
        Specified model input shape, values represent (height, width, channels) if not manually reconfigured
    name : str
        Custom name for the built model
    pooling : str or None
        What type of pooling to use in the model (avg/max/None)
    optimizer : str or tf.keras.optimizers
        Model optimizer (name or instance)
    metrics : list
        Metrics used by the model
    add_top : bool
        Whether we want to add custom top layers to our model

    Returns
    -------
    tf.keras.models.Model
        Built and compiled model instance

    Raises
    -------
    ValueError
        Given base_name argument doesn't match any of the available architectures
    """
    if base_name == 'VGG16':
        model_func = VGG16
    elif base_name == 'DenseNet169':
        model_func = DenseNet169
    elif base_name == 'ResNet50':
        model_func = ResNet50
    else:
        raise ValueError(f"Pre-built model {base_name} not available")

    base_model = model_func(include_top=False,
                            weights=weights,
                            input_shape=shape,
                            pooling=pooling)
    x = base_model.output

    if add_top:
        if pooling is None:
            x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)

    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input,
                  outputs=predictions,
                  name=name)

    model.compile(optimizer=optimizer,
                  loss=BinaryCrossentropy(),
                  metrics=metrics)
    return model


def create_generators(rotation_r=20, w_shift_r=0.05, h_shift_r=0.05, brightness_r=(0.9, 1.1), zoom_r=0.1, h_flip=True,
                      pre_func=rescale):
    """
    Creates two ImageDataGenerator instances for training and validation. Training generator uses several augmentation
    methods, while valid generator only used provided preprocessing function.

    More detailed information about ImageDataGenerator and parameters can be found in TensorFlow documentation:
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

    Parameters
    ----------
    rotation_r
        Range specifying rotation degree
    w_shift_r
        Defines image width shifting, one possible option is to provide fraction of image width
    h_shift_r
        Defines image height shifting, one possible option is to provide fraction of image height
    brightness_r
        Range from which brightness value will be picked, it can be both increased and decreased
    zoom_r
        Random zooming range
    h_flip
        Whether to flip the image horizontally or not
    pre_func
        Function applied to each input before feeding it to the model. It can be a custom-defined function, that
        takes the image as an argument and return the modified image of the same shape

    Returns
    -------
    Training data generator with augmentation and validation generator
    """
    train_gen = ImageDataGenerator(rotation_range=rotation_r,
                                   width_shift_range=w_shift_r,
                                   height_shift_range=h_shift_r,
                                   brightness_range=brightness_r,
                                   zoom_range=zoom_r,
                                   horizontal_flip=h_flip,
                                   fill_mode='reflect',
                                   preprocessing_function=pre_func)

    valid_gen = ImageDataGenerator(preprocessing_function=pre_func)
    return train_gen, valid_gen


def create_dataframe_flows(train_gen, valid_gen, train_df, valid_df, directory=ROOT_DIR, img_size=(224, 224), batch_size=16):
    """
    Creates train and validation dataset flows from provided dataframes.

    Parameters
    ----------
    train_gen : ImageDataGenerator
        Used for feeding training data to the model
    valid_gen : ImageDataGenerator
        Used for feeding validation data to the model
    train_df : pd.Dataframe
        Must contain columns['filepath', 'label'] with training image filepaths with their labels
    valid_df : pd.Dataframe
        Must contain columns['filepath', 'label'] with validation image filepaths with their labels
    directory : str
        Root directory of the dataset to be joined with filepaths obtained from dataframes
    img_size : Tuple(x, y)
        Image size required by the model input
    batch_size : int
        Number of images in each batch (it's best practice to use powers of 2)

    Returns
    -------
    Training and validation dataframe flows
    """
    train_flow = train_gen.flow_from_dataframe(dataframe=train_df,
                                               directory=directory,
                                               x_col='filepath',
                                               y_col='label',
                                               class_mode='binary',
                                               target_size=img_size,
                                               batch_size=batch_size,
                                               seed=27)

    valid_flow = valid_gen.flow_from_dataframe(dataframe=valid_df,
                                               directory=directory,
                                               x_col='filepath',
                                               y_col='label',
                                               class_mode='binary',
                                               target_size=img_size,
                                               batch_size=batch_size,
                                               seed=27)
    return train_flow, valid_flow


def get_class_weights(df):
    """
    Calculates class weights by basic approach for binary tasks:
    w0 = n_1 / n_samples
    w1 = n_0 / n_samples

    Parameters
    ----------
    df : pd.Dataframe
        Dataset containing filepaths to images together with labels

    Returns
    ----------
    Class weights as dictionary in format defined in TensorFlow documentation
    """
    n_samples = len(df.index)
    n_positive = len(df.loc[df['label'] == 'positive'].index)
    n_negative = len(df.loc[df['label'] == 'negative'].index)

    w_positive = n_negative / n_samples
    w_negative = n_positive / n_samples

    return {0: w_negative, 1: w_positive}

