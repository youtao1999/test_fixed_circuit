# using Pkg
# Pkg.activate("/home/youtao/CT_MPS_mini/CT")
using CT

# print out initial vector
using ITensors
using Random
using JSON
using HDF5
using ProgressMeter
using CT: U

function save_initial_mps(initial_mps)
    # Save MPS in HDF5 format (preserves all MPS structure)
    h5open("initial_state.h5", "w") do file
        write(file, "mps", initial_mps)
    end
end

function save_circuit(circuit)
    open("circuit.json", "w") do f
        write(f, JSON.json(circuit))
        println("circuit.json written")
    end
end

# generate list of gates
function generate_circuit(L, p_ctrl, p_proj, rng; save=true)
    circuit = []
    for t in 1:2*L^2
    # for t in 1:10
        if rand(rng) < p_ctrl
            push!(circuit, ["C", rand(rng)])
        else
            push!(circuit, ["B", U(4, rng)])
            if rand(rng) < p_proj
                # force the proejctive measurement to be onto 0 for now
                push!(circuit, ["P", 0, rand(rng)])
            end
        end
    end
    if save
        save_circuit(circuit)
        println("circuit saved")
    else
        println("circuit generated")
    end
    println(length(circuit))
    return circuit
end

# save_circuit(generate_circuit(p_ctrl, p_proj, rng))
function save_initial_state(initial_state)
    open("initial_state.json", "w") do f
        write(f, JSON.json(initial_state))
        println("initial_state.json written")
    end
end

function read_initial_state()
    return JSON.parsefile("initial_state.json")
end

function read_circuit()
    return JSON.parsefile("circuit.json")
end

function read_initial_mps()
    initial_state_loaded = h5open("initial_state.h5", "r") do file
        read(file, "mps", MPS)
    end
    return initial_state_loaded
end

function generate_config(ct, p_ctrl, p_proj)
    if isfile("initial_state.h5")
        rm("initial_state.h5")
    end
    save_initial_mps(ct.mps)
    save_initial_state(vec(array(contract(ct.mps))))
    if isfile("circuit.json")
        rm("circuit.json")
    end
    generate_circuit(ct.L, p_ctrl, p_proj, ct.rng, save=true)
end

function postprocess_bernoulli!(circuit)
    circuit_copy = []
    for (i, cir) in enumerate(circuit)
        if cir[1] == "B"
            # println(complex(cir[2][1][1]["re"], cir[2][1][1]["im"]))
            modified_cir = ["B", hcat([[complex(cir[2][i][j]["re"], cir[2][i][j]["im"]) for j in 1:4] for i in 1:4]...)]
            push!(circuit_copy, modified_cir)
        else
            push!(circuit_copy, cir)
        end
    end
    return circuit_copy
end

function main(L, ancilla, folded, seed_vec, _maxdim, _cutoff, p_ctrl, p_proj, x0)
    ct_f=CT.CT_MPS(L=L,seed=seed_vec,folded=folded,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([1//3,2//3]),_maxdim=_maxdim, _maxdim0=_maxdim, simplified_U=true,builtin=false)
    i = 1
    generate_config(ct_f, p_ctrl, p_proj)
    ct_f.mps = read_initial_mps()
    # println(vec(array(contract(ct_f.mps))))
    circuit = read_circuit()

    # postprocess bernoulli maps in circuit
    circuit = postprocess_bernoulli!(circuit)
    counter = 0
    for cir in circuit[1:2]
        counter += 1
        i = CT.random_control_fixed_circuit!(ct_f, i, cir)
        println(vec(array(contract(ct_f.mps)))[1:10])
        # normalize!(ct_f.mps)
    end
    return ct_f.mps
end

if abspath(PROGRAM_FILE) == @__FILE__
    # L = 8
    # seed_vec = 123457
    # folded = true
    # ancilla = 0
    # _maxdim = 2^9
    # p_ctrl = 0.5
    # p_proj = 0.5
    # ct_f=CT.CT_MPS(L=L,seed=seed_vec,folded=folded,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([1//3,2//3]),_maxdim=_maxdim, _maxdim0=_maxdim, simplified_U=true,builtin=false)
    # i = 1
    # generate_config(ct_f, p_ctrl, p_proj)
    main(8, 0, true, 123457, 2^9, 1e-15, 0.5, 0.5, nothing)
end