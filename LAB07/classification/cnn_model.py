from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(64, 64, 3), num_conv_blocks=2, initial_filters=32, dense_units=64):
    model = models.Sequential()
    
    # สร้าง Convolutional Layer และ MaxPooling
    for i in range(num_conv_blocks):
        filters = initial_filters * (2 ** i)
        if i == 0:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
    # Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  # ใช้ Sigmoid สำหรับ Binary Classification (Cat vs Dog)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model