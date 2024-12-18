import gc
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import datetime
import traceback
from utils import plot_confusion_matrix

# Algoritmo GABRA
def GABRA(partitions, gpu_capacities, max_generations=50):
    m = len(gpu_capacities)
    n = len(partitions)
    gpus = list(range(m))

    # Inicialización de la población
    P = P0(partitions, gpus)
    Z_best = None
    f_Z_best = float('inf')
    t = 0

    while t < max_generations:
        # Selección aleatoria de dos padres
        Y1, Y2 = random.sample(P, 2)

        # Cruce
        Y1_new, Y2_new = Ψc(Y1, Y2)

        # Mutación
        W1 = mutate(Y1_new, gpus)
        W2 = mutate(Y2_new, gpus)

        # Evaluación de fitness
        f_W1 = f(W1, partitions, gpu_capacities)
        f_W2 = f(W2, partitions, gpu_capacities)

        # Actualización del mejor fitness
        if f_W1 < f_Z_best:
            Z_best, f_Z_best = W1, f_W1
        if f_W2 < f_Z_best:
            Z_best, f_Z_best = W2, f_W2

        # Reemplazo en la población
        P.extend([W1, W2])

        t += 1

    return Z_best, f_Z_best

# Otras funciones auxiliares para GABRA
def P0(partitions, gpus):
    return [[random.choice(gpus) for _ in partitions] for _ in range(len(partitions))]

def Ψc(parent1, parent2):
    cp = len(parent1) // 2
    return parent1[:cp] + parent2[cp:], parent2[:cp] + parent1[cp:]

def mutate(solution, gpus):
    idx = random.randint(0, len(solution)-1)
    solution[idx] = random.choice(gpus)
    return solution

def f(W, partitions, gpu_capacities):
    load = [0] * len(gpu_capacities)
    for i, gpu in enumerate(W):
        load[gpu] += partitions[i]
    return max(load)

# Lectura de datos
gc.collect()
PATH_TO_REP = 'data/'
results_folder = 'result_resnet/'

metadata = pd.read_csv(PATH_TO_REP + 'metadata.csv')
testdata = pd.read_csv(PATH_TO_REP + 'test.csv')

LABEL_1, LABEL_2 = 'AD', 'Normal'
smc_mask = ((metadata.Label == LABEL_1) | (metadata.Label == LABEL_2)).values.astype('bool')
y = (metadata[smc_mask].Label == LABEL_1).astype(np.int32).values
data = np.zeros((smc_mask.sum(), 1, 110, 110, 110), dtype='float32')

for it, im in tqdm(enumerate(metadata[smc_mask].Path.values), total=smc_mask.sum(), desc='Cargando datos de entrenamiento'):
    mx = nib.load(im).get_data().max()
    data[it, 0, :, :, :] = np.array(nib.load(im).get_data()) / mx

X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)

# Entrenamiento con GABRA
gpu_capacities = [4, 4, 4]
partitions = [len(X_train) // len(gpu_capacities)] * len(gpu_capacities)
best_allocation, best_fitness = GABRA(partitions, gpu_capacities)

# Convertir los datos a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Crear un DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Definir el modelo en PyTorch
class ResNet3D(nn.Module):
    def __init__(self):
        super(ResNet3D, self).__init__()
        # Definir las capas de la red aquí (ejemplo)
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(16 * 55 * 55 * 55, 2)  # Ajusta esto según la arquitectura final

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 55 * 55 * 55)  # Flatten
        x = self.fc1(x)
        return x

model = ResNet3D()

# Definir los criterios y optimizadores
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000027)

# Entrenamiento del modelo
for epoch in range(80):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = []
        targets = []
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.numpy())
            targets.extend(labels.numpy())
        
        acc = accuracy_score(targets, preds)
        print(f'Epoch {epoch+1}, Validation Accuracy: {acc:.4f}')

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'models/resnet3d.pth')

# Evaluación final
model.eval()
preds, targ = [], []
with torch.no_grad():
    for img in X_val_tensor:
        out = model(img.unsqueeze(0))
        preds.append(out.argmax(axis=1).item())
        targ.append(y_val_tensor[it].item())

acc = accuracy_score(targ, preds)
print(f'\nTest Accuracy: {acc:.4f}')

# Generar matriz de confusión
plot_confusion_matrix(np.array(targ), np.array(preds), classes=['AD', 'Normal'], normalize=True)
plt.savefig(results_folder + 'Confusion_Matrix_Normalized.png')
