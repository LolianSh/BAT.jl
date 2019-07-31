#=
# Run with:
include("test_mcmc_rand.jl"); show_scatter()
=#

using Distributions
using PDMats
using StatsBase
using IntervalSets
#using Base.Test
#using BAT
using JLD
using Plots
#using Plotly

using BAT.Logging


#α_vec = collect(1:1:20)
#m_vec = collect(1:5:101)
#df_vec = collect(1.0:10.0:51.0)
#iter = collect(1:1:50)

α_vec = collect(0.2:0.2:1.0)
#m_vec = collect(1::33)
m_vec = [1, 3, 7, 15, 33]
#df_vec = collect(1.0:15.0:31.0)
df_vec = [0.1, 1.0, 10.0]
iter = collect(1:1:1)


n = size(iter, 1)

nsq = sqrt(n)

d_mean = load("data_mean.jld")["data_mean"]
d_cov = load("data_cov.jld")["data_cov"]
d_tuned = load("data_tuned.jld")["data_tuned"]


cumulative_tuned = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), 1)


m_mean = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), 2)
m_cov = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), 2, 2)

se_mean = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), 2)
se_cov = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), 2, 2)

for i in indices(α_vec, 1)
    for m in indices(m_vec, 1)
        for k in indices(df_vec, 1)
            m_mean[i, m, k, 1] = mean(d_mean[i, m, k, :, 1])
            m_mean[i, m, k, 2] = mean(d_mean[i, m, k, :, 2])

            se_mean[i, m, k, 1] = std(d_mean[i, m, k, :, 1])
            se_mean[i, m, k, 2] = std(d_mean[i, m, k, :, 2])

            m_cov[i, m, k, 1, 1] = mean(d_cov[i, m, k, :, 1, 1])
            m_cov[i, m, k, 1, 2] = mean(d_cov[i, m, k, :, 1, 2])
            m_cov[i, m, k, 2, 1] = mean(d_cov[i, m, k, :, 2, 1])
            m_cov[i, m, k, 2, 2] = mean(d_cov[i, m, k, :, 2, 2])

            se_cov[i, m, k, 1, 1] = std(d_cov[i, m, k, :, 1, 1])
            se_cov[i, m, k, 1, 2] = std(d_cov[i, m, k, :, 1, 2])
            se_cov[i, m, k, 2, 1] = std(d_cov[i, m, k, :, 2, 1])
            se_cov[i, m, k, 2, 2] = std(d_cov[i, m, k, :, 2, 2])

            cumulative_tuned[i, m, k, 1] = sum(d_tuned[i, m, k, :, 1])

        end
    end
end

best_mean_dim1 = minimum(se_mean[:, :, :, 1])
best_mean_dim2 = minimum(se_mean[:, :, :, 2])

ind_best_mean_dim1 = ind2sub(se_mean[:, :, :, 1], indmin(se_mean[:, :, :, 1]))
ind_best_mean_dim1 = ind2sub(se_mean[:, :, :, 1], indmin(se_mean[:, :, :, 1]))

best_cov_dim11 = minimum(se_cov[:, :, :, 1, 1])
best_cov_dim12 = minimum(se_cov[:, :, :, 1, 2])
best_cov_dim21 = minimum(se_cov[:, :, :, 2, 1])
best_cov_dim22 = minimum(se_cov[:, :, :, 2, 2])

ind_best_cov_dim11 = ind2sub(se_cov[:, :, :, 1, 1], indmin(se_cov[:, :, :, 1, 1]))
ind_best_cov_dim12 = ind2sub(se_cov[:, :, :, 1, 2], indmin(se_cov[:, :, :, 1, 2]))
ind_best_cov_dim21 = ind2sub(se_cov[:, :, :, 2, 1], indmin(se_cov[:, :, :, 2, 1]))
ind_best_cov_dim22 = ind2sub(se_cov[:, :, :, 2, 2], indmin(se_cov[:, :, :, 2, 2]))


Plots.heatmap(d_tuned[:, :, 1, 1, 1])
Plots.heatmap(se_mean[:, :, 1, 1])
Plots.histogram2d(d_mean[:, :, 1, 1, 1])
Plots.heatmap(se_mean[:, :, 1, 1], color=:inferno_r)
Plots.histogram(d_cov[5, 1, 1, :, 1, 1])
#[m_cov[:, 3, 1, 1, 1] se_cov[:, 3, 1, 1, 1]]
Plots.heatmap(cumulative_tuned[:, :, 1, 1])
