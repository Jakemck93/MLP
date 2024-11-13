from engine import Value
from  nn import MLP
import random

class Prediction:
    def __init__(self, n_features)
        self.model = MLP(n_features,[4,2,1])
        self.learning_rate = 0.01

    def predict(self, x):
        output = self.model(x)

        return self._sigmoid(output)
    
    def train(self, X, y, epochs=100):
        losses = []
        for epoch in range(epochs):
            total_loss = 0

            for xi, yi in zip(X, y):
                # Convert the input to Value objects
                x = [Value(xij) if not isinstance(xij, Value) else xij for xij in xi]
                #forward pass
                pred = self.predict(x)
                #Binary cross entropy loss
                loss = -yi * self.safe_log(pred) - (1 - yi) * self.safe_log(1 - pred)
                total_loss += loss.data
                #backprop
                self.model.zero_grad()
                loss.backward() 
                #update weights
                for p in self.model.parameters():
                    p.data -= self.learning_rate * p.grad
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        return losses
    def _sigmoid(self, x):
        return (1 + (-x).exp())**-1
    def _safe_log(self, x):
        eps = 1e-7
        return (x + eps).log()
        
    