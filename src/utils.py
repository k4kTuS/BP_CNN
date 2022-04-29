import pandas as pd
import tensorflow as tf

DATASET_PATHS = '../MURA-v1.1/detailed_paths.csv'


def get_dataframe(body_part, split):
    """
    Filters dataset by body part and data split

    Parameters
    ----------
    body_part : str
        Name of specific body part / all body parts (no filtering)
    split : str
        Name of specific data split / all images (no filtering)

    Returns
    -------
    pd.Dataframe
        Dataframe containing filtered dataset
    """
    if body_part == 'ALL':
        body_part = '.*'
    if split == 'ALL':
        split = '.*'

    df = pd.read_csv(DATASET_PATHS)
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
        model_func = tf.keras.applications.VGG16
    elif base_name == 'DenseNet169':
        model_func = tf.keras.applications.ResNet50
    elif base_name == 'ResNet50':
        model_func = tf.keras.applications.ResNet50
    else:
        raise ValueError(f"Pre-built model {base_name} not available")

    base_model = model_func(include_top=False,
                            weights=weights,
                            input_shape=shape,
                            pooling=pooling)
    x = base_model.output
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=base_model.input,
                                  outputs=predictions,
                                  name=name)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model
