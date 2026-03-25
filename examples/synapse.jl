mutable struct Synapse
    inidx::Int
    outidx::Int
    w::Float64
    wmax::Float64
    pretrace::Float64
    posttrace::Float64
    learningrate::Float64
end


function update_plasticity!(synapse::Synapse, dt::Float64, τ::Float64, input::Bool, output::Bool)
    decay = exp(-dt / τ)
    synapse.pretrace *= decay
    synapse.posttrace *= decay

    if input 
        synapse.pretrace += 1.0
        synapse.w -= synapse.posttrace * synapse.learningrate * (synapse.w / synapse.wmax)
    end

    if output
        synapse.posttrace += 1.0
        synapse.w += synapse.pretrace * synapse.learningrate * (1.0 - synapse.w / synapse.wmax)
    end

    synapse.w = max(0.0, min(synapse.w, synapse.wmax))
end

function test_stdp()
    dt = 1.0
    τ = 20.0

    syn = Synapse(1, 2, 0.5, 1.0, 0.0, 0.0, 0.01)

    for t in 1:5000

        input = (t % 10 == 0)
        output = (t % 10 == 2) || (rand() < 0.05)

        update_plasticity!(syn, dt, τ, input, output)

        if t % 50 == 0
            println("t=$t | w=$(syn.w)")
        end
    end
end

function test_competition()
    dt = 1.0
    τ = 20.0

    synA = Synapse(1, 3, 0.5, 1.0, 0.0, 0.0, 0.01)
    synB = Synapse(2, 3, 0.5, 1.0, 0.0, 0.0, 0.01)

    for t in 1:3000

        inputA = (t % 10 == 0)          
        inputB = (rand() < 0.1)         

        output = (t % 10 == 2)

        update_plasticity!(synA, dt, τ, inputA, output)
        update_plasticity!(synB, dt, τ, inputB, output)

        if t % 100 == 0
            println("t=$t | wA=$(synA.w) | wB=$(synB.w)")
        end
    end

    println("\nFinal:")
    println("Synapse A (correlated): ", synA.w)
    println("Synapse B (random): ", synB.w)
end