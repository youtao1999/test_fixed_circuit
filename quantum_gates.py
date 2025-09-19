from sv_fixed_circuit import read_circuit, read_initial_state
import sys
sys.path.append("/home/youtao/CT_toy")
import numpy as np
# from QCT_util import U_simp, U
from QCT_fixed_circuit import QCT_fixed_circuit

def postprocess_bernoulli(circuit):
    circuit_copy = []
    for cir in circuit:
        if cir[0] == "B":
            matrix = np.array([[complex(cir[1][i][j]["re"], cir[1][i][j]["im"]) for j in range(4)] for i in range(4)])
            modified_cir = ["B", matrix]
            circuit_copy.append(modified_cir)
        else:
            circuit_copy.append(cir)
    return circuit_copy

if __name__ == "__main__":
    initial_state = read_initial_state()
    circuit = read_circuit()
    # print(circuit[0])
    circuit_postprocessed = postprocess_bernoulli(circuit)
    # print(circuit_postprocessed[0])
    # print([cir[0] for cir in circuit_postprocessed])
    # print(circuit_postprocessed[2])
    # print(initial_state.flatten())
    qct = QCT_fixed_circuit(
        circuit=circuit_postprocessed,
        initial_state=initial_state,
    )
    for i in range(len(circuit_postprocessed))[0:1]:
        state, _ = qct.step_evolution(i)
        # print(state.flatten()[0:9])
        # print(qct.state.flatten()[0:9])
    # print(qct.state.flatten())