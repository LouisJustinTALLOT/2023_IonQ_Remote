import qiskit
import numpy as np
from qiskit.circuit.library.standard_gates import RYGate

NB_QUBITS = 5

# Build a 3 quibit circuit
circuit = qiskit.QuantumCircuit(NB_QUBITS)

# Apply a Hadamard gate to all qubits except the last one
for i in range(NB_QUBITS - 1):
    circuit.h(i)

# Add a controlled RY gate to the last qubit
gate = RYGate(2 * np.pi).control(num_ctrl_qubits=2)
circuit.append(gate, [0, 1, NB_QUBITS - 1])

print(circuit)