from gradlite.models.mlp import MLP


if __name__ == "__main__":

    X = [[2.0, 3.0, -1.0],
         [3.0, -1.0, 0.5],
         [0.5, 1.0, 1.0],
         [1.0, 1.0, -1.0]]

    Y = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1], ['tanh', 'tanh', 'tanh'])
    model.summary()

    learning_rate = 0.2
    NUM_EPOCHS = 1000

    for i in range(NUM_EPOCHS):
        # Forward Pass
        Y_hat = [model(x) for x in X]

        # Calculate Loss per sample                                                                                                               
        loss = [(y_hat - y)**2 for y, y_hat in zip(Y, Y_hat)]

        # Calculate the cost
        cost = sum(loss)

        print(f"Epoch {i}: Total Loss: {float(cost.data)}")

        # Clear residual Gradients
        for p in model.parameters:
            p.grad = 0.0

        # The Magic !!! Backpropagation ✨
        cost.backward()

        # Parameter Updation
        for p in model.parameters:
            p.data -= learning_rate * p.grad

    print([model(x).data for x in X])