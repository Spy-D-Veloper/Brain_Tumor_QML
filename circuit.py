# ============================================================
# STEP 0: LOAD DATA + CREATE LABEL
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pca_df = pd.read_csv("preprocessed/features_pca.csv")
raw_df = pd.read_csv("preprocessed/features_raw.csv")

data = pca_df.merge(
    raw_df[["subject_id", "ratio_et_wt"]],
    on="subject_id",
    how="left"
)

median_val = data["ratio_et_wt"].median()
data["label"] = (data["ratio_et_wt"] >= median_val).astype(int)

X = data.drop(columns=["subject_id", "ratio_et_wt", "label"])
y = data["label"]

print("✅ Data Loaded")
print("Class Distribution:\n", y.value_counts())


# ============================================================
# STEP 4: QUANTUM CIRCUIT (ONLY RX, RY, RZ)
# ============================================================
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

NUM_QUBITS = 4

def create_circuit(features, theta):
    qc = QuantumCircuit(NUM_QUBITS)

    for i in range(NUM_QUBITS):
        qc.ry(features[i], i)
        qc.rz(theta[i], i)
        qc.rx(theta[i]/2, i)

    for i in range(NUM_QUBITS - 1):
        qc.cx(i, i+1)

    qc.measure_all()
    return qc


def optimize_circuit(qc):
    return transpile(qc, optimization_level=2)


# ============================================================
# STEP 5: NOISE MODEL
# ============================================================
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, ReadoutError

def create_noise_model():
    noise_model = NoiseModel()

    dep1 = depolarizing_error(0.01, 1)
    dep2 = depolarizing_error(0.02, 2)

    bit_flip = pauli_error([('X', 0.05), ('I', 0.95)])

    meas_error = ReadoutError([[0.9, 0.1], [0.1, 0.9]])

    noise_model.add_all_qubit_quantum_error(dep1, ['ry','rz','rx'])
    noise_model.add_all_qubit_quantum_error(dep2, ['cx'])
    noise_model.add_all_qubit_quantum_error(bit_flip, ['ry','rz','rx'])
    noise_model.add_all_qubit_readout_error(meas_error)

    return noise_model


def run_circuit(qc, noise_model=None):
    sim = AerSimulator(noise_model=noise_model)
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=512).result()
    return result.get_counts()


# ============================================================
# STEP 6: NOISE MITIGATION (ZNE → MEM)
# ============================================================

def zne_mitigate(counts_clean, counts_noisy):
    total_clean = sum(counts_clean.values())
    total_noisy = sum(counts_noisy.values())

    mitigated = {}

    for state in counts_clean.keys():
        p_clean = counts_clean.get(state, 0) / total_clean
        p_noisy = counts_noisy.get(state, 0) / total_noisy

        p_zne = 2 * p_clean - p_noisy
        mitigated[state] = max(p_zne, 0)

    return mitigated


def measurement_error_mitigation(counts):
    total = sum(counts.values())
    mitigated = {}

    for state, prob in counts.items():
        corrected = (prob * 0.9) + (0.1 / len(counts))
        mitigated[state] = corrected

    return mitigated


# ============================================================
# STEP 7: HYBRID OPTIMIZATION
# ============================================================

noise_model = create_noise_model()
theta = np.random.rand(NUM_QUBITS)

epochs = 5
lr = 0.1

results = []

def bit_to_class(bitstring):
    return int(bitstring, 2) % 2


for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        features = X.iloc[i].values[:NUM_QUBITS]
        label = y.iloc[i]

        qc = create_circuit(features, theta)
        qc_opt = optimize_circuit(qc)

        counts_clean = run_circuit(qc_opt)
        counts_noisy = run_circuit(qc_opt, noise_model)

        # Before mitigation
        s_clean = max(counts_clean, key=counts_clean.get)
        s_noisy = max(counts_noisy, key=counts_noisy.get)

        # ZNE → MEM
        zne_counts = zne_mitigate(counts_clean, counts_noisy)
        final_counts = measurement_error_mitigation(zne_counts)

        s_final = max(final_counts, key=final_counts.get)

        pred = bit_to_class(s_final)

        # 🔥 LOSS
        loss = (pred - label) ** 2
        total_loss += loss

        # 🔥 HYBRID UPDATE
        theta = theta - lr * (pred - label)

        # Show only 5 circuits (first epoch)
        if epoch == 0 and i < 5:
            print(f"\nSample {i}")
            print(qc_opt.draw())
            print("Clean:", s_clean, "(", counts_clean[s_clean], ")",
                  "| Noisy:", s_noisy, "(", counts_noisy[s_noisy], ")",
                  "| Final:", s_final)

        # Save only final epoch
        if epoch == epochs - 1:
            results.append({
                "true": int(label),
                "pred": int(pred)
            })

    print(f"\nEpoch {epoch+1} Loss: {total_loss:.4f}")


# ============================================================
# STEP 8: METRICS + GRAPH
# ============================================================
# ============================================================
# STEP 8: FULL METRICS + ROBUSTNESS ANALYSIS
# ============================================================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

clean_preds = []
noisy_preds = []
mitigated_preds = []

for i in range(len(X)):
    features = X.iloc[i].values[:NUM_QUBITS]

    qc = create_circuit(features, theta)
    qc_opt = optimize_circuit(qc)

    # Run circuits
    counts_clean = run_circuit(qc_opt)
    counts_noisy = run_circuit(qc_opt, noise_model)

    # Dominant states
    s_clean = max(counts_clean, key=counts_clean.get)
    s_noisy = max(counts_noisy, key=counts_noisy.get)

    # Mitigation
    zne_counts = zne_mitigate(counts_clean, counts_noisy)
    final_counts = measurement_error_mitigation(zne_counts)
    s_final = max(final_counts, key=final_counts.get)

    # Convert to class
    clean_preds.append(bit_to_class(s_clean))
    noisy_preds.append(bit_to_class(s_noisy))
    mitigated_preds.append(bit_to_class(s_final))


# ============================================================
# METRIC FUNCTION
# ============================================================

def compute_metrics(name, preds):
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print(f"\n{name}")
    print("----------------------")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return acc


# ============================================================
# PRINT ALL METRICS
# ============================================================

print("\n==============================")
print("🔷 FULL QUANTUM COMPARISON")
print("==============================")

acc_clean = compute_metrics("Quantum (No Noise)", clean_preds)
acc_noisy = compute_metrics("Quantum (With Noise)", noisy_preds)
acc_mitigated = compute_metrics("Quantum (Mitigated)", mitigated_preds)


# ============================================================
# ROBUSTNESS ANALYSIS
# ============================================================

drop = acc_clean - acc_noisy
recovery = acc_mitigated - acc_noisy

print("\n==============================")
print("🔷 ROBUSTNESS ANALYSIS")
print("==============================")
print(f"Accuracy Drop (Noise Effect)   : {drop:.4f}")
print(f"Recovery (Mitigation Benefit)  : {recovery:.4f}")


# ============================================================
# GRAPH SAVE (UPDATED)
# ============================================================

labels = ["No Noise", "With Noise", "Mitigated"]
values = [acc_clean, acc_noisy, acc_mitigated]

plt.figure()
plt.bar(labels, values)
plt.ylim(0,1)
plt.title("Quantum Robustness Comparison")
plt.savefig("results/quantum_comparison.png")
plt.close()

print("✅ Comparison graph saved to results/quantum_comparison.png")