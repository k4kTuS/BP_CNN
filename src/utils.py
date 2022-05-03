import pandas as pd
from tensorflow.keras.applications import DenseNet169, ResNet50, VGG16
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.metrics import CohenKappa

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


def build_model(base_name, weights, shape, name, pooling='max', optimizer=Adam(learning_rate=0.0001), metrics=[CohenKappa(num_classes=2, name='kappa')]):
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
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input,
                  outputs=predictions,
                  name=name)

    model.compile(optimizer=optimizer,
                  loss=BinaryCrossentropy,
                  metrics=metrics)
    return model


def create_generators(rotation_r=20, w_shift_r=0.05, h_shift_r=0.05, brightness_r=(0.9, 1.1), zoom_r=0.1, h_flip=True,
                      pre_func=None):
    """
    Creates ImageDataGenerator instances for training and validation

    Parameters
    ----------
    rotation_r
    w_shift_r
    h_shift_r
    brightness_r
    zoom_r
    h_flip
    pre_func

    Returns
    -------

    """
    train_gen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=rotation_r,
                                   width_shift_range=w_shift_r,
                                   height_shift_range=h_shift_r,
                                   brightness_range=brightness_r,
                                   zoom_range=zoom_r,
                                   horizontal_flip=h_flip,
                                   fill_mode='reflect',
                                   preprocessing_function=pre_func)

    valid_gen = ImageDataGenerator(rescale=1. / 255,
                                   preprocessing_function=pre_func)
    return train_gen, valid_gen


def create_dataframe_flows(train_gen, valid_gen, train_df, valid_df, directory=ROOT_DIR, img_size=(224, 224), batch_size=16):
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


def get_class_weights(df, n_classes=2):
    """
    Calculates "balanced" class weights for given dataset using following formula:
    wi = n_samples / (n_classes * n_i)

    Parameters
    ----------
    df : pd.Dataframe
        Dataset containing filepaths to images together with labels
    n_classes: int
        Number of classes in the dataset

    Returns
    ----------
    Class weights as dictionary in format defined in TensorFlow documentation
    """
    n_samples = len(df.index)
    n_positive = len(df.loc[df['label'] == 'positive'].index)
    n_negative = len(df.loc[df['label'] == 'negative'].index)

    w_positive = n_samples / (n_classes * n_positive)
    w_negative = n_samples / (n_classes * n_negative)

    return {0: w_negative, 1: w_positive}

