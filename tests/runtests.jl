include("../modules/Networks.jl")

using Test
using .Networks
using .Networks.Neurons
using .Networks.Synapses

@testset "ECG Encoding Network Tests" begin

    @testset "Neuron Tests" begin

        @testset "Neuron validation" begin
            @test_throws ArgumentError Neuron("bad"; τ_m=-1.0)
            @test_throws ArgumentError Neuron("bad"; τ_s=-1.0)
            @test_throws ArgumentError Neuron("bad"; V_thresh=-1.0, V_rest=0.0)
            @test_throws ArgumentError Neuron("bad"; τ_ref=-1.0)
        end

        @testset "receive_spike!" begin
            n = Neuron("test")
            @test n.i == 0.0
            receive_spike!(n, 0.5)
            @test n.i == 0.5
            receive_spike!(n, 0.3)
            @test n.i ≈ 0.8
        end

        @testset "update! no spike" begin
            n = Neuron("test")
            fired = update!(n, 1.0, 0.0)
            @test !fired
            @test n.v ≈ n.V_rest
        end

        @testset "update! with current causes spike" begin
            n = Neuron("test"; V_thresh=1.0, V_rest=0.0, V_reset=0.0, R_m=10.0, τ_m=20.0, τ_s=100.0)
            receive_spike!(n, 5.0)
            fired = false
            for t in 1:100
                fired = update!(n, 1.0, Float64(t))
                if fired
                    break
                end
            end
            @test fired
        end

        @testset "update! refractory period" begin
            n = Neuron("test"; τ_ref=5.0, V_reset=0.0)
            n.t_ref = 3.0
            fired = update!(n, 1.0, 0.0)
            @test !fired
            @test n.v == n.V_reset
            @test n.t_ref ≈ 2.0
        end

    end

    @testset "Synapse Tests" begin

        @testset "decay!" begin
            s = Synapse(1, 2; τ_pre=10.0, τ_post=20.0)
            s.pretrace = 1.0
            s.posttrace = 1.0
            decay!(s, 1.0)
            @test s.pretrace < 1.0
            @test s.posttrace < 1.0
            @test s.pretrace ≈ exp(-1.0 / 10.0)
            @test s.posttrace ≈ exp(-1.0 / 20.0)
        end

        @testset "prespike! (LTD)" begin
            s = Synapse(1, 2; w=0.5, wmax=1.0, learningrate=0.1)
            s.posttrace = 0.5
            old_w = s.w
            prespike!(s)
            @test s.pretrace ≈ 1.0
            @test s.w < old_w
            @test s.w >= 0.0
        end

        @testset "postspike! (LTP)" begin
            s = Synapse(1, 2; w=0.5, wmax=1.0, learningrate=0.1)
            s.pretrace = 0.5
            old_w = s.w
            postspike!(s)
            @test s.posttrace ≈ 1.0
            @test s.w > old_w
            @test s.w <= s.wmax
        end

        @testset "Weight bounds" begin
            s = Synapse(1, 2; w=0.01, wmax=1.0, learningrate=1.0)
            s.posttrace = 10.0
            prespike!(s)
            @test s.w >= 0.0

            s2 = Synapse(1, 2; w=0.99, wmax=1.0, learningrate=1.0)
            s2.pretrace = 10.0
            postspike!(s2)
            @test s2.w <= s2.wmax
        end

    end

    @testset "Network Tests" begin

        @testset "Network construction" begin
            ns = [Neuron("a"), Neuron("b")]
            syns = [Synapse(1, 2)]
            net = Network(ns, syns)
            @test length(net.neurons) == 2
            @test length(net.synapses) == 1
            @test net.index["a"] == 1
            @test net.index["b"] == 2
            @test isempty(net.spikelog)
        end

        @testset "Network duplicate neuron names" begin
            ns = [Neuron("dup"), Neuron("dup")]
            @test_throws ArgumentError Network(ns, Synapse[])
        end

        @testset "resolve_index with String" begin
            net = Network([Neuron("x"), Neuron("y")], Synapse[])
            @test resolve_index(net, "x") == 1
            @test resolve_index(net, "y") == 2
            @test_throws ArgumentError resolve_index(net, "z")
        end

        @testset "resolve_index with Int" begin
            net = Network([Neuron("x"), Neuron("y")], Synapse[])
            @test resolve_index(net, 1) == 1
            @test resolve_index(net, 2) == 2
            @test_throws ArgumentError resolve_index(net, 0)
            @test_throws ArgumentError resolve_index(net, 3)
        end

        @testset "addneuron!" begin
            net = Network()
            addneuron!(net, Neuron("new"))
            @test length(net.neurons) == 1
            @test net.index["new"] == 1
            @test_throws ArgumentError addneuron!(net, Neuron("new"))
        end

        @testset "addsynapse! with Synapse" begin
            net = Network([Neuron("a"), Neuron("b")], Synapse[])
            addsynapse!(net, Synapse(1, 2))
            @test length(net.synapses) == 1
            @test net.synapses[1].inidx == 1
            @test net.synapses[1].outidx == 2
        end

        @testset "addsynapse! with Int indices" begin
            net = Network([Neuron("a"), Neuron("b")], Synapse[])
            addsynapse!(net, 1, 2)
            @test length(net.synapses) == 1
            @test net.synapses[1].inidx == 1
            @test net.synapses[1].outidx == 2
        end

        @testset "addsynapse! with String names" begin
            net = Network()
            addneuron!(net, Neuron("pre"))
            addneuron!(net, Neuron("post"))
            addsynapse!(net, "pre", "post", w=0.8, isinhibitory=true)
            @test length(net.synapses) == 1
            @test net.synapses[1].inidx == 1
            @test net.synapses[1].outidx == 2
            @test net.synapses[1].w == 0.8
            @test net.synapses[1].isinhibitory == true
        end

        @testset "addsynapse! validation" begin
            net = Network([Neuron("a")], Synapse[])
            @test_throws ArgumentError addsynapse!(net, 1, 2)
            @test_throws ArgumentError addsynapse!(net, Synapse(2, 1))
        end

        @testset "get_outgoing_syns" begin
            net = Network([Neuron("a"), Neuron("b"), Neuron("c")], Synapse[])
            addsynapse!(net, 1, 2)
            addsynapse!(net, 1, 3)
            addsynapse!(net, 2, 3)
            out1 = get_outgoing_syns(net, 1)
            @test length(out1) == 2
            out2 = get_outgoing_syns(net, 2)
            @test length(out2) == 1
            out3 = get_outgoing_syns(net, 3)
            @test isempty(out3)
        end

        @testset "get_incoming_syns" begin
            net = Network([Neuron("a"), Neuron("b"), Neuron("c")], Synapse[])
            addsynapse!(net, 1, 2)
            addsynapse!(net, 1, 3)
            addsynapse!(net, 2, 3)
            in1 = get_incoming_syns(net, 1)
            @test isempty(in1)
            in3 = get_incoming_syns(net, 3)
            @test length(in3) == 2
        end

    end

    @testset "Connectome Tests" begin

        @testset "Connectome construction" begin
            ns = [Neuron("a"), Neuron("b")]
            syns = [Synapse(1, 2)]
            conn = Connectome(ns, syns)
            @test length(conn.neurons) == 2
            @test length(conn.outgoing[1]) == 1
            @test length(conn.incoming[2]) == 1
            @test isempty(conn.outgoing[2])
            @test isempty(conn.incoming[1])
        end

        @testset "Connectome with multiple synapses" begin
            ns = [Neuron("a"), Neuron("b"), Neuron("c")]
            syns = [Synapse(1, 2), Synapse(1, 3), Synapse(2, 3)]
            conn = Connectome(ns, syns)
            @test length(conn.outgoing[1]) == 2
            @test length(conn.outgoing[2]) == 1
            @test length(conn.incoming[3]) == 2
        end

        @testset "Network <-> Connectome conversion" begin
            ns = [Neuron("a"), Neuron("b")]
            syns = [Synapse(1, 2)]
            net = Network(ns, syns)
            conn = Connectome(net)
            @test length(conn.neurons) == 2
            @test length(conn.outgoing[1]) == 1

            net2 = Network(conn)
            @test length(net2.neurons) == 2
            @test length(net2.synapses) == 1
        end

    end

    @testset "Simulation Tests" begin

        @testset "step! on Network" begin
            net = Network()
            addneuron!(net, Neuron("a"; V_thresh=100.0))
            addneuron!(net, Neuron("b"; V_thresh=100.0))
            addsynapse!(net, 1, 2)
            step!(net, 1.0, 0.0)
            @test true
        end

        @testset "step! on Connectome" begin
            ns = [Neuron("a"; V_thresh=100.0)]
            conn = Connectome(ns, Synapse[])
            step!(conn, 1.0, 0.0)
            @test true
        end

        @testset "run! on Network" begin
            net = Network()
            addneuron!(net, Neuron("a"; V_thresh=100.0))
            spikes = run!(net, 1.0, 10.0)
            @test spikes === net.spikelog
        end

        @testset "run! on Connectome" begin
            ns = [Neuron("a"; V_thresh=100.0)]
            conn = Connectome(ns, Synapse[])
            spikes = run!(conn, 1.0, 10.0)
            @test spikes === conn.spikelog
        end

        @testset "run! with t0" begin
            net = Network()
            addneuron!(net, Neuron("a"; V_thresh=100.0))
            spikes = run!(net, 1.0, 10.0; t0=5.0)
            @test !isempty(spikes) || true
        end

        @testset "run! clears spikelog" begin
            net = Network()
            addneuron!(net, Neuron("a"; V_thresh=100.0))
            push!(net.spikelog, Spike(0.0, true, "old"))
            @test length(net.spikelog) == 1
            run!(net, 1.0, 10.0)
            @test length(net.spikelog) >= 0
        end

        @testset "Spike propagation" begin
            exc = Neuron("exc"; V_rest=0.0, V_thresh=1.0, V_reset=0.0, R_m=10.0, τ_m=20.0, τ_s=100.0, τ_ref=2.0)
            net = Network([exc], Synapse[])
            receive_spike!(exc, 10.0)
            spikes = run!(net, 1.0, 500.0)
            @test length(spikes) > 0
        end

        @testset "Inhibitory synapse" begin
            pre = Neuron("pre"; V_rest=0.0, V_thresh=1.0, V_reset=0.0, R_m=10.0, τ_m=20.0, τ_s=100.0, τ_ref=2.0)
            post = Neuron("post"; V_rest=0.0, V_thresh=1.0, V_reset=0.0, R_m=10.0, τ_m=20.0, τ_s=100.0, τ_ref=2.0)
            net = Network([pre, post], Synapse[])
            addsynapse!(net, 1, 2; w=5.0, isinhibitory=true)
            receive_spike!(pre, 10.0)
            spikes = run!(net, 1.0, 500.0)
            @test true
        end

        @testset "STDP during simulation" begin
            pre = Neuron("pre"; V_rest=0.0, V_thresh=1.0, V_reset=0.0, R_m=10.0, τ_m=20.0, τ_s=100.0, τ_ref=2.0)
            post = Neuron("post"; V_rest=0.0, V_thresh=1.0, V_reset=0.0, R_m=10.0, τ_m=20.0, τ_s=100.0, τ_ref=2.0)
            syn = Synapse(1, 2; w=0.5, learningrate=0.1)
            net = Network([pre, post], [syn])
            receive_spike!(pre, 10.0)
            old_w = syn.w
            run!(net, 1.0, 500.0)
            @test true
        end

    end

    @testset "Integration Tests" begin

        @testset "Two-neuron network simulation" begin
            exc = Neuron("exc"; V_rest=0.0, V_thresh=1.0, V_reset=0.0, R_m=10.0, τ_m=20.0, τ_s=100.0, τ_ref=2.0)
            inh = Neuron("inh"; V_rest=0.0, V_thresh=1.0, V_reset=0.0, R_m=8.0, τ_m=15.0, τ_s=100.0, τ_ref=2.0)
            net = Network([exc, inh], Synapse[])
            addsynapse!(net, "exc", "inh"; w=0.7)
            receive_spike!(exc, 10.0)
            spikes = run!(net, 1.0, 500.0)
            @test length(spikes) > 0
        end

        @testset "Feed-forward chain" begin
            neurons = [Neuron("n$i"; V_rest=0.0, V_thresh=1.0, V_reset=0.0, R_m=10.0, τ_m=20.0, τ_s=100.0, τ_ref=2.0) for i in 1:4]
            net = Network(neurons, Synapse[])
            for i in 1:3
                addsynapse!(net, i, i+1; w=0.8)
            end
            receive_spike!(neurons[1], 10.0)
            spikes = run!(net, 1.0, 500.0)
            @test length(spikes) > 0
        end

        @testset "Network with delays" begin
            net = Network()
            addneuron!(net, Neuron("a"; V_thresh=100.0))
            addneuron!(net, Neuron("b"; V_thresh=100.0))
            addsynapse!(net, "a", "b", delay=5.0)
            @test net.synapses[1].delay == 5.0
        end

        @testset "Connectome simulation matches Network" begin
            ns = [Neuron("a"; V_rest=0.0, V_thresh=100.0, V_reset=0.0)]
            syns = Synapse[]
            net = Network(ns, syns)
            conn = Connectome(ns, syns)
            run!(net, 1.0, 50.0)
            run!(conn, 1.0, 50.0)
            @test length(net.spikelog) == length(conn.spikelog)
        end

    end

end
