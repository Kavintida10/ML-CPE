from sklearn.neural_network import MLPClassifier

def build_nn_model(hidden_units=(128, 64), max_iter=30, learning_rate=0.001):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_units,
        activation='relu',
        solver='adam',
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        random_state=42,
        early_stopping=False
    )
    return model