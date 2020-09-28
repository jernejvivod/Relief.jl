
# Test functionality with continuous features.
@testset "Evaporative Cooling ReliefF - Continuous Features" begin
    data = rand(1000, 10)
    for idx1 = 1:size(data, 2) - 1
        for idx2 = idx1+1:size(data, 2)
            target = Int64.(data[:, idx1] .> data[:, idx2])
            rank = Relief.ecrelieff(data, target, f_type="continuous")
            @test Set([rank[idx1], rank[idx2]]) == Set([1, 2])
        end
    end
end


# Test functionality with discrete features.
@testset "Evaporative Cooling ReliefF - Discrete Features" begin
    data = rand([0, 1, 2, 3], 1000, 10)
    for idx1 = 1:size(data, 2) - 1
        for idx2 = idx1+1:size(data, 2)
            target = Int64.(data[:, idx1] .> data[:, idx2])
            rank = Relief.ecrelieff(data, target, f_type="discrete")
            @test Set([rank[idx1], rank[idx2]]) == Set([1, 2])
        end
    end
end

