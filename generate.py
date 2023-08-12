from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.optimizer import Adam
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

IPython.display.Audio('30s_seq.wav')
X, Y, n_values, indices_values, chords = load_music_utils('original_metheny.mid')
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('shape of X:', X.shape)
print('Shape of Y:', Y.shape)
print('Number of chords', len(chords))

n_a = 64 # number of dimensions for the hidden state of each LSTM cell.
n_values = 90 # number of music values

class DJModel(nn.Module):
    def __init__(self, Tx, n_a, n_values):
        super(DJModel, self).__init__()
        self.Tx = Tx
        self.n_a = n_a
        self.n_values = n_values
        self.LSTM_cell = nn.LSTM(n_values, n_a, batch_first=True)
        self.densor = nn.Linear(n_a, n_values)

    def forward(self, X, a0, c0):
        outputs = []
        a = a0.unsqueeze(0).contiguous()  # Add an extra dimension to a0
        c = c0.unsqueeze(0).contiguous()  # Add an extra dimension to c0

        for t in range(self.Tx):
            x = X[:, t, :]
            x = x.view(-1, 1, self.n_values)
            a, (a, c) = self.LSTM_cell(x, (a, c))
            out = self.densor(a.view(-1, self.n_a))
            outputs.append(out)

        return outputs


# Create the model
Tx = 30
model = DJModel(Tx, n_a, n_values)

# UNIT TEST
print(model)

# optimizer
opt = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Input data and initial states
X = torch.randn(60, 30, 90)
Y = torch.randn(30, 60, 90)
m = 60
a0 = torch.zeros(m, n_a)
c0 = torch.zeros(m, n_a)

# Training loop
epochs = 100
losses = []

for epoch in range(epochs):
    model.train()
    outputs = model(X, a0, c0)
    loss = 0

    for t in range(Tx):
        output_t = outputs[t]
        target_t = Y[t]
        loss += loss_fn(output_t, target_t)

    loss.backward()
    opt.step()
    opt.zero_grad()
    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Loss at epoch {epoch + 1}: {loss.item()}")

# Plot the loss
import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

LSTM_cell=nn.LSTM(n_values, n_a, batch_first=True)
densor = nn.Linear(n_a, n_values)

import torch
import torch.nn as nn

class music_inference_model(nn.Module):
    def __init__(self, LSTM_cell, densor, Ty=100):
        super(music_inference_model, self).__init__()
        self.LSTM_cell = LSTM_cell
        self.densor = densor
        self.Ty = Ty

    def forward(self, x0, a0, c0):
        n_values = self.densor.out_features
        n_a = self.LSTM_cell.hidden_size

        a = a0.unsqueeze(0)
        c = c0.unsqueeze(0)
        x = x0

        outputs = []

        for t in range(self.Ty):
            a, (a, c) = self.LSTM_cell(x, (a, c))

            out = self.densor(a.squeeze(1))
            outputs.append(out.unsqueeze(1))
            x = torch.argmax(out, dim=-1)
            x = torch.nn.functional.one_hot(x, num_classes=n_values)
            x = x.unsqueeze(0).float()

        output_sequence = torch.cat(outputs, dim=1)
        return output_sequence
      
inference_model = music_inference_model(LSTM_cell,densor, Ty = 50)
print(inference_model)

x_initializer = np.zeros((1, 1, n_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

x_initializer = torch.tensor(x_initializer)
a_initializer = torch.tensor(a_initializer)
c_initializer = torch.tensor(c_initializer)

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, c_initializer = c_initializer):
    n_values = x_initializer.shape[2]
    
    if len(x_initializer.shape) == 2:
        x_initializer = x_initializer.unsqueeze(0)
    if len(a_initializer.shape) == 1:
        a_initializer = a_initializer.unsqueeze(0)
    if len(c_initializer.shape) == 1:
        c_initializer = c_initializer.unsqueeze(0)
    
    with torch.no_grad():
        pred = inference_model(x_initializer, a_initializer, c_initializer)
    
    indices = np.argmax(pred, axis = -1)
    print(indices.shape)
    indices=torch.as_tensor(indices)
    results = torch.nn.functional.one_hot(indices, n_values)
    
    return results, indices
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))

out_stream = generate_music(inference_model, indices_values, chords)

mid2wav('my_music.midi')
IPython.display.Audio('rendered.wav')

IPython.display.Audio('30s_trained_model.wav')
