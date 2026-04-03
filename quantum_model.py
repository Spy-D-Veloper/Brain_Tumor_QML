from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import pandas as pd

# ===============================
# 1️⃣ Load PCA Correctly
# ===============================

pca_path = "preprocessed/features_pca.csv"

# Force comma separator
X = pd.read_csv(pca_path, sep=",")

# Drop the first column (subject_id)
X = X.drop(columns=["subject_id"])

# Now take first sample
x = X.iloc[0].values

n_qubits = 4
x = x[:n_qubits]

print("Real PCA Sample Used:", x)

# ===============================
# 2️⃣ Build Quantum Circuit
# ===============================

qc = QuantumCircuit(n_qubits)

# Angle encoding (PCA values)
for i in range(n_qubits):
    qc.ry(float(x[i]), i)

# Trainable parameters
theta = np.random.rand(n_qubits)
for i in range(n_qubits):
    qc.ry(theta[i], i)

# Entanglement
for i in range(n_qubits - 1):
    qc.cx(i, i+1)

# Measurement
qc.measure_all()

print(qc.draw())

# ===============================
# 3️⃣ Step 5: Noise Modeling
# ===============================

noise_model = NoiseModel()

measurement_error = pauli_error([('X', 0.03), ('I', 0.97)])
noise_model.add_all_qubit_quantum_error(measurement_error, ['measure'])

sim = AerSimulator(noise_model=noise_model)
# ===============================
# 4️⃣ Run Simulation with Noise
# ===============================

sim = AerSimulator(noise_model=noise_model)

result = sim.run(qc, shots=1024).result()
counts = result.get_counts()

print("Measurement Results (With Noise):", counts)