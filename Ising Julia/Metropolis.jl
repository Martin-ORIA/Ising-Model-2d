using BenchmarkTools, StaticArrays
using Random: default_rng, seed!
using Base.Threads
using LoopVectorization
using Plots
using Profile

function ising2d_ifelse!(s, β, rng=default_rng())
    m, n = size(s)
    min_h = -4
    max_h = 4
    prob = [1/(1+exp(-2*β*h)) for h in min_h:max_h]
    magnetization = 0.0
    for j in 1:n 
        for i in 1:m
            NN = s[ifelse(i == 1, m, i-1), j]   #voisins
            SS = s[ifelse(i == m, 1, i+1), j]
            WW = s[i, ifelse(j == 1, n, j-1)]
            EE = s[i, ifelse(j == n, 1, j+1)]
            h = NN + SS + WW + EE
            s[i,j] = ifelse(rand(rng) < prob[h-min_h+1], +1, -1)
        end
    end
    return s
end

const β_crit = log(1+sqrt(2))/2

rand_ising2d(m, n=m) = rand(Int8[-1, 1], m, n)


temperatures = range(0, stop=0.88, length=20)

magnetizations = zeros(20)
energie = zeros(20)
const niter = 10000 #pas de monte carlo
const neq = 10000   #pas d'équilibre
@time for (i, β) in enumerate(temperatures)
    s = rand_ising2d(20)   #taille de la matrice ici : 200
    @fastmath @threads for eq in 1:neq
        s = ising2d_ifelse!(s, β, default_rng())
    end
     @fastmath for iter in 1:niter
        s = @profile ising2d_ifelse!(s, β, default_rng())
        magnetizations[i] += abs(sum(s)) / (size(s,1)*size(s,2))
        energie[i] += (-sum(s.*(circshift(s, (-1,0)) + circshift(s, (1,0)) + circshift(s, (0,-1)) + circshift(s, (0,1)))))/4
    end
    magnetizations[i] = mean(magnetizations[i])
    energie[i] = mean(energie[i])
end




plot(temperatures, magnetizations, xlabel="Temperature", ylabel="Magnetization", seriestype=:scatter, show=true)
readline()