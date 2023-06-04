
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense,  GlobalAveragePooling2D


def Create(Trained_CNN_Model_dir):

    # load a pretrained model
    model = load_model(Trained_CNN_Model_dir, compile=False)

    # -- find index of the GlobalAveragePooling2D (avg_pool) layer
    i = 0
    for layer in model.layers:
        if isinstance(layer, GlobalAveragePooling2D):
            avg_pool_layer_index = i
        i = i + 1;
    # ---
    vg_pool = model.layers[avg_pool_layer_index].output
    output = Dense(1, activation='sigmoid', name='Output')(vg_pool)  # softmax

    model = Model(inputs=model.input, outputs=output)

    for layer in model.layers[:]:
        layer.trainable = True

    model.save('temp.h5')

    return model


