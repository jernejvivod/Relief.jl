"""
    square_to_vec(i::Array{Int64,1}, j::Array{Int64,1}, n::Int64)::Array{Int64,1}

Convert square pairwise distance matrix indices to distance vector indices.

Author: Jernej Vivod
"""
function square_to_vec(i::Array{Int64,1}, j::Array{Int64,1}, n::Int64)::Array{Int64,1}
    diag_msk = i .== j
    swp_msk = i .< j
    swp_i = i[swp_msk]
    i[swp_msk] = j[swp_msk] 
    j[swp_msk] = swp_i
    res = n .* j .- j .* ((j .+ 1) ./ 2) .+ i .- 1 .- j
    res[diag_msk] .= -1
    return res
end

