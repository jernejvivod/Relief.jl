
# Test functionality with continuous features.
@testset "TuRF - Continuous Features" begin
    data = rand(1000, 10)
    for idx1 = 1:size(data, 2) - 1
        for idx2 = idx1+1:size(data, 2)
            target = Int64.(data[:, idx1] .> data[:, idx2])
            weights = Relief.turf(data, target, 5, rba=Relief.relieff)
            @test all(weights[idx1] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])
            @test all(weights[idx2] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])
        end
    end
end


# Test functionality with discrete features.
@testset "TuRF - Discrete Features" begin
    data = rand([0, 1, 2, 3], 1000, 10)
    for idx1 = 1:size(data, 2) - 1
        for idx2 = idx1+1:size(data, 2)
            target = Int64.(data[:, idx1] .> data[:, idx2])
            weights = Relief.turf(data, target, 5, rba=Relief.relieff)
            @test all(weights[idx1] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])
            @test all(weights[idx2] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])
        end
    end
end


# Test functionality with default RBA (ReliefF).
@testset "TuRF - Default RBA" begin
    data = rand([0, 1, 2, 3], 1000, 10)
    idx1, idx2 = 1, 2
    target = Int64.(data[:, idx1] .> data[:, idx2])
    weights = Relief.turf(data, target, 5)
    @test all(weights[idx1] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])
    @test all(weights[idx2] .>= weights[(1:end .!= idx1) .& (1:end .!= idx2)])

end

