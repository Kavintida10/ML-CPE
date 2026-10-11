import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, MobileNet, EfficientNetB0, VGG16

def build_model(model_name="mobilenet_v2", num_classes=5, input_shape=(128, 128, 3)):
    if model_name == "mobilenet_v2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "mobilenet_v1":
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "efficientnet_b0":
        inputs = layers.Input(shape=input_shape)
        x = layers.Rescaling(255.0)(inputs)
        eff_base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)
        eff_base.trainable = False
        x = layers.GlobalAveragePooling2D()(eff_base.output)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    elif model_name == "vgg16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model