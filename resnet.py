# ============================================================
# STEP 0: LOAD DATA + PCA (99% VARIANCE)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca_df = pd.read_csv("preprocessed/features_pca.csv")
raw_df = pd.read_csv("preprocessed/features_raw.csv")

data = pca_df.merge(
    raw_df[["subject_id", "ratio_et_wt"]],
    on="subject_id",
    how="left"
)

median_val = data["ratio_et_wt"].median()
data["label"] = (data["ratio_et_wt"] >= median_val).astype(int)

# Features + Labels
X_full = data.drop(columns=["subject_id", "ratio_et_wt", "label"])
y = data["label"]

# 🔥 PCA with 99% variance
pca = PCA(n_components=0.99)
X_pca = pca.fit_transform(X_full)

print("Original Features:", X_full.shape[1])
print("Reduced Features:", X_pca.shape[1])

X = pd.DataFrame(X_pca)

# Shuffle + Split
data_final = X.copy()
data_final["label"] = y.values
data_final = data_final.sample(frac=1, random_state=42).reset_index(drop=True)

train_data = data_final.iloc[:180]
test_data  = data_final.iloc[180:230]

X_train = train_data.drop(columns=["label"])
y_train = train_data["label"].reset_index(drop=True)

X_test = test_data.drop(columns=["label"])
y_test = test_data["label"].reset_index(drop=True)


# ============================================================
# STEP 1: IMPORTS
# ============================================================
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, ReadoutError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# STEP 2: PARAMETERS
# ============================================================
NUM_QUBITS = 4
NUM_LAYERS = 4   # 🔥 increased
SHOTS = 512
BATCH_SIZE = 32
EPOCHS = 15


# ============================================================
# STEP 3: NORMALIZATION (NUMPY SAFE)
# ============================================================
X_min = X_train.to_numpy().min(axis=0)
X_max = X_train.to_numpy().max(axis=0)

def encode_features(x):
    x = np.asarray(x, dtype=float).flatten()
    x_norm = (x - X_min[:NUM_QUBITS]) / (X_max[:NUM_QUBITS] - X_min[:NUM_QUBITS] + 1e-8)
    x_norm = np.clip(x_norm, 0, 1)
    return x_norm * np.pi


# ============================================================
# STEP 4: IMPROVED QUANTUM CIRCUIT
# ============================================================
def create_circuit(features, theta):
    qc = QuantumCircuit(NUM_QUBITS)

    features = np.asarray(features, dtype=float).flatten()[:NUM_QUBITS]
    features = encode_features(features)

    idx = 0
    for _ in range(NUM_LAYERS):

        # Data encoding
        for i in range(NUM_QUBITS):
            qc.ry(features[i], i)
            qc.rz(features[i], i)

        # Trainable block
        for i in range(NUM_QUBITS):
            qc.rx(theta[idx], i)
            qc.ry(theta[idx+1], i)
            qc.rz(theta[idx+2], i)
            idx += 3

        # 🔥 Ring entanglement
        for i in range(NUM_QUBITS):
            qc.cx(i, (i+1) % NUM_QUBITS)

        # 🔥 Full entanglement
        for i in range(NUM_QUBITS):
            for j in range(i+1, NUM_QUBITS):
                qc.cz(i, j)

    qc.measure_all()
    return qc


def optimize_circuit(qc):
    return transpile(qc, optimization_level=2)


def run_circuit(qc, noise_model=None):
    sim = AerSimulator(noise_model=noise_model)
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=SHOTS).result()
    return result.get_counts()


# ============================================================
# STEP 5: LOSS
# ============================================================
def get_prob_class1(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.5
    prob = 0
    for bitstring, c in counts.items():
        if bitstring[-1] == '1':
            prob += c / total
    return prob


def compute_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y*np.log(p) + (1-y)*np.log(1-p))


# ============================================================
# STEP 6: SPSA OPTIMIZER
# ============================================================
def spsa_step(theta, features, label, lr, c):
    delta = 2*np.random.randint(0, 2, size=len(theta)) - 1

    theta_plus  = theta + c * delta
    theta_minus = theta - c * delta

    qc_p = optimize_circuit(create_circuit(features, theta_plus))
    qc_m = optimize_circuit(create_circuit(features, theta_minus))

    p_p = get_prob_class1(run_circuit(qc_p))
    p_m = get_prob_class1(run_circuit(qc_m))

    loss_p = compute_loss(p_p, label)
    loss_m = compute_loss(p_m, label)

    grad = (loss_p - loss_m) / (2 * c * delta)
    grad = np.clip(grad, -5, 5)

    theta = theta - lr * grad
    return theta, float((loss_p + loss_m)/2)


# ============================================================
# STEP 7: TRAINING
# ============================================================
theta_size = NUM_LAYERS * NUM_QUBITS * 3
theta = np.random.uniform(0, 2*np.pi, theta_size)

initial_lr = 0.02
c = 0.05

for epoch in range(EPOCHS):
    total_loss = 0
    lr = initial_lr / (1 + 0.1 * epoch)

    indices = np.random.permutation(len(X_train))

    for start in range(0, len(X_train), BATCH_SIZE):
        batch_idx = indices[start:start+BATCH_SIZE]

        batch_loss = 0

        for i in batch_idx:
            features = X_train.iloc[i].to_numpy().astype(float)[:NUM_QUBITS]
            label = int(y_train.iloc[i])

            theta, loss = spsa_step(theta, features, label, lr, c)
            batch_loss += loss

        total_loss += batch_loss / len(batch_idx)

    print(f"Epoch {epoch+1} | LR: {lr:.4f} | Loss: {total_loss:.4f}")


# ============================================================
# STEP 8: NOISE + MITIGATION
# ============================================================
def create_noise_model():
    noise_model = NoiseModel()

    dep1 = depolarizing_error(0.02, 1)
    dep2 = depolarizing_error(0.05, 2)

    bit_flip = pauli_error([('X', 0.05), ('I', 0.95)])
    meas_error = ReadoutError([[0.9, 0.1], [0.1, 0.9]])

    noise_model.add_all_qubit_quantum_error(dep1, ['ry','rz','rx'])
    noise_model.add_all_qubit_quantum_error(dep2, ['cx'])
    noise_model.add_all_qubit_quantum_error(bit_flip, ['ry','rz','rx'])
    noise_model.add_all_qubit_readout_error(meas_error)

    return noise_model


def zne_mitigate(clean, noisy):
    total_c = sum(clean.values())
    total_n = sum(noisy.values())

    if total_c == 0 or total_n == 0:
        return clean

    mitigated = {}
    for s in clean:
        pc = clean.get(s,0)/total_c
        pn = noisy.get(s,0)/total_n
        mitigated[s] = max(2*pc - pn, 0)

    if sum(mitigated.values()) == 0:
        mitigated = {s:1 for s in clean}

    return mitigated


def measurement_error_mitigation(counts):
    total = sum(counts.values())
    if total == 0:
        return counts
    return {s:0.85*(c/total) + 0.15/len(counts) for s,c in counts.items()}


# ============================================================
# STEP 9: EVALUATION
# ============================================================
noise_model = create_noise_model()

clean_preds, noisy_preds, mitigated_preds = [], [], []

for i in range(len(X_test)):
    features = X_test.iloc[i].to_numpy().astype(float)[:NUM_QUBITS]

    qc = optimize_circuit(create_circuit(features, theta))

    counts_clean = run_circuit(qc)
    counts_noisy = run_circuit(qc, noise_model)

    zne = zne_mitigate(counts_clean, counts_noisy)
    final = measurement_error_mitigation(zne)

    p_clean = get_prob_class1(counts_clean)
    p_noisy = get_prob_class1(counts_noisy)
    p_final = get_prob_class1(final)

    clean_preds.append(int(p_clean > 0.5))
    noisy_preds.append(int(p_noisy > 0.5))
    mitigated_preds.append(int(p_final > 0.5))


# ============================================================
# STEP 10: METRICS
# ============================================================
def metrics(name, preds):
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"\n{name}")
    print("------------------------------")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")


print("\n==============================")
print("🔷 QUANTUM RESULTS (TEST SET)")
print("==============================")

metrics("Quantum (Clean)", clean_preds)
metrics("Quantum (Noise)", noisy_preds)
metrics("Quantum (Mitigated)", mitigated_preds)