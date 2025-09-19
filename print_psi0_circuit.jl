# using Pkg
# Pkg.activate("/home/youtao/CT_MPS_mini/CT")
using CT

# print out initial vector
using ITensors
using Random
using JSON
using HDF5
using ProgressMeter

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
            push!(circuit, ["C"])
        else
            push!(circuit, ["U", rand(rng, 12)...])
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


function main(L, ancilla, folded, seed_vec, _maxdim, _cutoff, p_ctrl, p_proj, x0)

    ct_f=CT.CT_MPS(L=L,seed=seed_vec,folded=folded,store_op=false,store_vec=false,ancilla=ancilla,debug=false,xj=Set([1//3,2//3]),_maxdim=_maxdim, _maxdim0=_maxdim, simplified_U=true,builtin=false)
    i = 1
    qubit_site, ram_phy, phy_ram, phy_list = CT._initialize_basis(L, ancilla, folded)
    rng = MersenneTwister(seed_vec)
    if isfile("initial_state.h5")
        rm("initial_state.h5")
    end
    save_initial_mps(ct_f.mps)
    if isfile("circuit.json")
        rm("circuit.json")
    end
    generate_circuit(L, p_ctrl, p_proj, rng, save=true)
    ct_f.mps = read_initial_mps()
    circuit = read_circuit()
    counter = 0
    for cir in circuit
        # println(i, " ", cir[1])
        counter += 1
        println(counter, " maxrss: ", Sys.maxrss() / 1024^2, " MB, heap: ", Base.gc_live_bytes() / 1024^2, " MB, op: ", cir[1])
        # println(cir[1])
        i = CT.random_control_fixed_circuit!(ct_f, i, cir)
    end

    return ct_f.mps
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(20, 0, true, 123457, 50, 1e-15, 0.4, 0.7, nothing)
end