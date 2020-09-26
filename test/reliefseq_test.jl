using Relief, Test

# Test functionality with continuous features.
@testset "ReliefSeq - Continuous Features" begin
    data = rand(500, 10)
    for idx1 = 1:size(data, 2) - 1
        for idx2 = idx1+1:size(data, 2)
            target = Int64.(data[:, idx1] .> data[:, idx2])
            weights = Relief.reliefseq(data, target, mode="k_nearest", f_type="continuous")
            @test all(weights[idx1] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])
            @test all(weights[idx2] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])
        end
    end
end


# Test functionality with discrete features.
@testset "ReliefSeq - Discrete Features" begin
    data = rand([0, 1, 2, 3], 500, 10)
    for idx1 = 1:size(data, 2) - 1
        for idx2 = idx1+1:size(data, 2)
            target = Int64.(data[:, idx1] .> data[:, idx2])
            weights = Relief.reliefseq(data, target, mode="k_nearest", f_type="discrete")
            @test all(weights[idx1] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])
            @test all(weights[idx2] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])
        end
    end
end

