# using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn
using PyCall

const naive_bayes = PyNULL()
const model_selection = PyNULL()
function __init__()
    copy!(naive_bayes, pyimport("sklearn.naive_bayes"))
    copy!(model_selection, pyimport("sklearn.model_selection"))
end

"""
    ec_ranking(data::Array{<:Real,2}, target::Array{<:Integer,1}, 
                    weights::Array{<:AbstractFloat,1}, mu_vals::Array{<:AbstractFloat,1})::Array{Int64,1}

Perform evaporative feature ranking (auxiliary function).
"""
function ec_ranking(data::Array{<:Real,2}, target::Array{<:Integer,1}, 
                    weights::Array{<:AbstractFloat,1}, mu_vals::Array{<:AbstractFloat,1})::Array{Int64,1}

	# Get maximal ReliefF weight and compute epsilon values.
	max_weight = maximum(weights)
	eps = (max_weight .- weights)./max_weight

	# Set initial tinfo value.
	tinfo = 1

	# Initialize vector of feature ranks.
	rank = Array{Int64}(undef, length(weights))

	# Initialize vector of indices in original weights array.
	index_vector = collect(1:length(weights))
	
	# Initialize initial rank value.
	rank_value_next = length(weights)
	
	# Initialize variable that holds current best tinfo value.
	best_tinfo = tinfo

	# While there are unranked features, perform evaporation.
	@inbounds while length(index_vector) > 1
		

		### Grid search for best temperature from previous step ###


		# Initialize variables that hold current maximal CV score 
		# and index of removed feature.
		max_cv_val = 0.0
		idx_removed_feature = -1

		
		# Perform grid search for best value of tinfo.
		@inbounds for tinfo_nxt = tinfo-0.3:0.1:tinfo+0.3

			# Compute weights for next value of tinfo.
			weights_nxt = eps .- tinfo_nxt.*mu_vals

			# Rank features.
			enumerated_weights = [weights_nxt collect(1:length(weights_nxt))]'
			rank_weights = zeros(Int64, length(weights_nxt))
			s = enumerated_weights[:, sortperm(enumerated_weights[1,:], rev=false)]
			rank_weights[convert.(Int64, s[2, :])] = 1:length(weights_nxt)


			# Remove lowest ranked feature.
			msk_rm = rank_weights .!= length(weights_nxt)

			# Perform 5-fold cross validation.
			data_filt = data[:, msk_rm]

			# Compare to current maximal CV value.
			cv_val_nxt = Statistics.mean(model_selection.cross_val_score(naive_bayes.GaussianNB(), data_filt, target; cv=5))

			# If CV value greater than current maximal, save current
			# CV score, tinfo and feature weights.
			if cv_val_nxt > max_cv_val
				# Set current tinfo value as best.
				best_tinfo = tinfo_nxt
				# Get index of removed feature.
				idx_removed_feature = findall(.!msk_rm)[1]
			end
		end
	
		# Remove evaporated feature from data matrix,
		# and data at corresponding index from eps vector and 
		# mu values vector.
		data = data[:, 1:end .!= idx_removed_feature]
		eps = eps[1:end .!= idx_removed_feature]
		mu_vals = mu_vals[1:end .!= idx_removed_feature]


		# Get index of evaporated feature in original data matrix.
		rank_index_next = index_vector[idx_removed_feature]

		# Delete removed feature from vector of indices.
		index_vector = deleteat!(index_vector, idx_removed_feature)

		# Assign rank to removed feature.
		rank[rank_index_next] = rank_value_next

		# Decrease next rank value.
		rank_value_next -= 1

	end
	
	# Assign final rank.
	rank[index_vector[1]] = rank_value_next

	# Return vector of ranks.
	return rank

end


"""
    function entropy(distribution::Array{<:Real,2})::Float64

Compute entropy of distribution (auxiliary function).
"""
function entropy(distribution::Array{<:Real,1})::Float64
    counts_classes = [sum(distribution .== el) for el in unique(distribution)]
    p_classes = counts_classes/length(distribution)
    return sum(p_classes.*log.(p_classes))
end


"""
    function joint_entropy_pair(distribution1::Array{<:Real,2}, distribution2::Array{<:Real,2})::Float64

Compute joint entropy of two distributions (auxiliary function).
"""
function joint_entropy_pair(distribution1::Array{<:Real,1}, distribution2::Array{<:Real,1})::Float64
    stacked = hcat(distribution1, distribution2)
    counts_pairs = [sum(all(stacked' .== reshape(el, length(el), 1), dims=1)) for el in eachrow(unique(stacked, dims=1))]
    p_pairs = counts_pairs/length(distribution1)
    return sum(p_pairs.*log.(p_pairs))
end


"""
    scaled_mutual_information(distribution1::Array{<:Real,2}, distribution2::Array{<:Real,2})::Float64

Compute scaled mutual information of two distributions (auxiliary function).
"""
function scaled_mutual_information(distribution1::Array{<:Real,1}, distribution2::Array{<:Real,1})::Float64
    return (entropy(distribution1) + entropy(distribution2) - joint_entropy_pair(distribution1, distribution2))/entropy(distribution1)
end


"""
    mu_vals(data::Array{<:Real,2}, target::Array{<:Integer,1})

Compute mu values from data and target (auxiliary function).
"""
function mu_vals(data::Array{<:Real,2}, target::Array{<:Integer,1})
    mu_vals = Array{Float64}(undef, size(data, 2))
    @inbounds for (idx, col) in enumerate(eachcol(data))
        mu_vals[idx] = scaled_mutual_information(Array{Real,1}(col), target)
    end 
    return mu_vals
end


"""
function ecrelieff(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                    k::Integer=5, dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2))::Array{Int64,1}

Compute feature rankings using Evaporative Cooling ReliefF algorithm.

---
# Reference:
- B.A. McKinney, D.M. Reif, B.C. White, J.E. Crowe, Jr., J.H. Moore.
Evaporative cooling feature selection for genotypic data involving interactions.
"""
function ecrelieff(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                    k::Integer=5, dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
                    mode::String="k_nearest", sig::Real=1.0, f_type::String="continuous")::Array{Int64,1}
    
    # Compute ReliefF weights.
    relieff_weights = relieff(data, target, m, k, dist_func, mode=mode, sig=sig, f_type=f_type)

    # Perform evaporative cooling feature selection to get feature ranks.
    return ec_ranking(data, target, relieff_weights, mu_vals(data, target))
end
