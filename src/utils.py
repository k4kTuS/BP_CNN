import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DEFAULT_DATASET_PATH = '/content/MURA-v1.1/tvt_detailed_paths.csv'
DEFAULT_DIRECTORY = '/content'


def get_dataframe(body_part, split, path=DEFAULT_DATASET_PATH):
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
    filtered_df = df[(df.body_part.str.match(body_part, case=False)) & (df.split.str.match(split, case=False))]
    return filtered_df


def build_model(base_name, weights, shape, pooling, name, optimizer, loss, metrics):
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
    pooling : str or None
        What type of pooling to use in the model (avg/max/None)
    name : str
        Custom name for the built model
    optimizer : str or tf.keras.optimizers
        Model optimizer (name or instance)
    loss : str or tf.keras.losses.Loss
        Name or instance of model loss function
    metrics : list
        Metrics used by the model

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
    elif base_name == 'DenseNet121':
        model_func = DenseNet121
    elif base_name == 'ResNet50':
        model_func = ResNet50
    else:
        raise ValueError(f"Pre-built model {base_name} not available")

    base_model = model_func(include_top=False,
                            weights=weights,
                            input_shape=shape,
                            pooling=pooling)
    x = base_model.output
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input,
                  outputs=predictions,
                  name=name)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model


def create_generators(rotation_r=20, w_shift_r=0.05, h_shift_r=0.05, brightness_r=(0.9, 1.1), zoom_r=0.1, h_flip=False):
    train_gen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=rotation_r,
                                   width_shift_range=w_shift_r,
                                   height_shift_range=h_shift_r,
                                   brightness_range=brightness_r,
                                   zoom_range=zoom_r,
                                   horizontal_flip=h_flip,
                                   fill_mode='reflect')

    valid_gen = ImageDataGenerator(rescale=1. / 255)
    return train_gen, valid_gen


def create_dataframe_flows(train_gen, valid_gen, body_part='ALL', directory=DEFAULT_DIRECTORY, dataset_type='default',
                           img_size=(224, 224), batch_size=32):
    train_df = get_dataframe(body_part, 'train')
    valid_df = get_dataframe(body_part, 'valid')

    if dataset_type == 'clahe':
        directory = directory + '/CLAHE/'

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
