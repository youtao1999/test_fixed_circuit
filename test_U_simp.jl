using Random
using LinearAlgebra
using JSON

# Add the CT module path and import it
push!(LOAD_PATH, "/home/youtao/CT_MPS_mini/CT/src")
using CT: U_simp

# Include the file that contains the read_circuit function
include("mps_fixed_circuit.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    circuit = read_circuit()
    for cir in circuit
        if cir[1] == "B"
            # println(cir[2:end])
            U_simp_example = U_simp(true, nothing, cir[2:end])
            println(U_simp_example)
            break
        end
    end
end