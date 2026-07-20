module Jswift
using DataFrames, CSV
using Random, StatsBase, Distributions
using Plots, StatsPlots, Measures
using Turing, Distributed, MCMCChains
using Serialization

"""
module structure:

export plustwo

plustwo(x) = return x+2
"""

export LLswift2
export execsacc
export lexrate
export LL
export swiftgen


function execsacc(kpos, k, next_tar, view, lbord, rbord, wlen, mu, sig, siglike)
    """ This is the execsacc function.

    # Arguments
    * `kpos` : current absolute letter position
    * `k` : current word (int)
    * `next_tar` : word chosen for next target (int)
    * `view` : center positions (letter) of all words in sentence
    * `lbord` : left border positions (letter) of all words in sentence
    * `rbord` : right border positions (letter) of all words in sentence
    * `wlen` : length of word k
    * `mu` : vector of means for each saccade type
    * `sig` : vector of sigmas for each saccade type
    * `siglike` : standard deviation of likelihood function

    # Returns
    * `{mu_p, sigma_p}` : parameters to determine saccadic landing position

    # Notes
    * Notes can go here

    # Examples
    ```julia
    julia> 
    ```
    """
    NW = length(view)

    # Determine intended saccade distance
    dist = view[next_tar] - kpos

    # 1. Determine saccade type 
    type = 1              # default: forward saccade
    if next_tar > k + 1
        type = 2          # skipping saccade
    end
    if next_tar == k
        if dist >= 0
            type = 3      # forward refixation
        else
            type = 4      # backward/regressive refixation
        end
    end
    if next_tar < k
        type = 5          # regressive saccade
    end

    # 2. Determine parameters
    mu_forw = kpos + mu[1]
    sig_forw = sig[1]
    sig_like_forw = siglike

    mu_skip = kpos + mu[2]
    sig_skip = sig[2]
    sig_like_skip = siglike

    mu_ref_forw = kpos + mu[3]
    sig_ref_forw = sig[3]
    sig_like_ref_forw = siglike

    mu_ref_regr = kpos + mu[4]
    sig_ref_regr = sig[4]
    sig_like_ref_regr = siglike

    mu_regr = kpos + mu[5]
    sig_regr = sig[5]
    sig_like_regr = siglike

    # 3. Bayesian posterior for landing-position density
    if type == 1
        mu_p = (sig_forw^2 * view[next_tar] + sig_like_forw^2 * mu_forw) / (sig_forw^2 + sig_like_forw^2)
        sigma_p = sqrt(1 / (1 / sig_forw^2 + 1 / sig_like_forw^2))
    end
    if type == 2
        mu_p = (sig_skip^2 * view[next_tar] + sig_like_skip^2 * mu_skip) / (sig_skip^2 + sig_like_skip^2)
        sigma_p = sqrt(1 / (1 / sig_skip^2 + 1 / sig_like_skip^2))
    end
    if type == 3 || type == 4
        p_forw = (wlen[k] - (kpos - lbord[k])) / wlen[k]
        if rand() > p_forw
            mu_tar = (kpos + rbord[k]) / 2
            mu_p = (sig_ref_forw^2 * mu_tar + sig_like_ref_forw^2 * mu_ref_forw) / (sig_ref_forw^2 + sig_like_ref_forw^2)
            sigma_p = sqrt(1 / (1 / sig_ref_forw^2 + 1 / sig_like_ref_forw^2))
        else
            mu_tar = (lbord[k] + kpos) / 2
            mu_p = (sig_ref_regr^2 * mu_tar + sig_like_ref_regr^2 * mu_ref_regr) / (sig_ref_regr^2 + sig_like_ref_regr^2)
            sigma_p = sqrt(1 / (1 / sig_ref_regr^2 + 1 / sig_like_ref_regr^2))
        end
    end
    if type == 5
        mu_p = (sig_regr^2 * view[next_tar] + sig_like_regr^2 * mu_regr) / (sig_regr^2 + sig_like_regr^2)
        sigma_p = sqrt(1 / (1 / sig_regr^2 + 1 / sig_like_regr^2))
    end

    # return parameters for saccadic landing position
    return [mu_p, sigma_p]
end


function lexrate(kpos, k, NW, view, wlen, δₗ, δᵣ, ν, κ)
    """ Computes the lexical processing rates for  each word of the sentence.
    
    # Arguments
    * `kpos` : current absolute letter position (in letters)
    * `k` : current word's position in sentence (in words)
    * `NW` : word chosen for next target (in words)
    * `view` : center positions of all words in sentence (in letters)
    * `wlen` : length of word k
    * `δₗ` : leftward processing span (in letters)
    * `δᵣ` : rightward processing span (in letters)
    * `ν` : shape of processing rate function
    * `κ` : word-length exponent   ########################### TODO: OR global inhibition? (see datagen)

    # Returns
    * `procrate` : processing rates of each word in sentence

    # Notes
    * Notes can go here

    # Examples
    ```julia
    julia> 
    ```
    """

    # parameters of the inverted quadratic function
    c₀ = (ν+1) ./ (ν .* (δₗ .+ δᵣ))            

    # calculating lexical processing rates
    T = typeof(δᵣ)
    procrate = zeros(T, NW)
    for j = 1:NW
        for l = 1:wlen[j]
            ε = view[j] - wlen[j] / 2 + l - 0.5 - kpos 
            prate = 0.0
            if -δₗ < ε && ε < 0  
                prate = c₀ .* (1 .- (abs(ε) ./ δₗ) .^ ν)       
            elseif ε >= 0 && ε < δᵣ  
                prate = c₀ .* (1 .- (abs(ε) ./ δᵣ) .^ ν)          
            end
            procrate[j] += prate
        end
        procrate[j] *= exp(-κ * log(wlen[j]))
    end
    procrate = procrate/sum(procrate)
    return procrate
end


function LLswift2(lfreq, pred, wlen, view, lbord, rbord, fword, fletter, fdur, δₗ, δᵣ, ν, μₜ, ι, β)
    """ Computes the loglikelihood for each word of the sentence.
    
    # Arguments
    * `lfreq` : log frequency of each word in sentence
    * `pred` : predictability of each word in sentence
    * `wlen` : length of word k
    * `view` : center positions of all words in sentence (in letters)
    * `lbord` : left border positions of all words in sentence (in letters)
    * `rbord` : right border positions of all words in sentence (in letters)
    * `fword` : sequence of fixated words in sentence (in words)
    * `fletter` : sequence of fixated letter in sentence (in letters)
    * `fdur` : fixation duration
    * `δₗ` : leftward processing span (in letters)
    * `δᵣ` : rightward processing span (in letters)
    * `ν` : shape of processing rate function
    * `μ` : ####### TODO: should be included in oculo -> pass oculo!!
    * `ι` : coupling of saccade timer and activation
    * `β` : word-frequency effect

    # Returns
    * `ll` : loglikelihood of sentence

    # Notes
    * Notes can go here

    # Examples
    ```julia
    julia> 
    ```
    """

    # TODO: extract fixed parameters from file!
    T = typeof(δᵣ)                    # variable type
    # fixed parameters
    η = T(-12.0)                     # noise in saccade targeting (log10)
    γ = T(1.0)                       # saccade targeting exponent
    ω = T(0.0)                       # decay of activation
    κ = T(0.0)                       # word-length exponent
    r = T(10.0)                      # overall processing rate
    ρ = T(7.0)                       # shape parameter of gamma distribution
    siglike = 2
 
    # unpack oculo parameters
    mu = oculo.mu
    sig = oculo.sigma

    # numbers of words and fixations
    NW = length(lfreq)
    Nfix = length(fletter)

    # define variables
    amax = one(T) .- β .* lfreq      # maximum activations
    a = zeros(T, NW)                 # word activations
    s = zeros(T, NW)                 # word saliencies
    p = zeros(T, NW)                 # selection probabilities
    λ = zeros(T,NW)                  # processing rates

    # additional parameters
    scale = μₜ / ρ                   # scale parameter for gamma density (mean μₜ, shape ρ)

    # initialize variables
    LLtime = 0.0
    LLspat = 0.0

    # set first fixation
    tfix = fdur[1]
    k = fword[1]
    kpos = fletter[1]
    next_kpos = fletter[2]
    next_k = fword[2]

    # loop over fixations
    for j in 1:(Nfix-2)

        # 1. Update processing rates
        λ = lexrate(kpos, k, NW, view, wlen, δₗ, δᵣ, ν, κ)
        #for j = 1:NW
        #    if a[j] > 0.5 * amax[j] && λ[j] < ω
        #        λ[j] = ω
        #    end
        #end

        # 2. Evolve activations
        a += r * λ * tfix / 1000.0
        for k in 1:NW
            a[k] = clamp(a[k], 0, amax[k])
        end
        # saliences
        s = amax .* sin.(π * a ./ amax) .+ 10.0^η

        # Compute selection probability
        p .= s .^ γ ./ sum(s .^ γ)

        # 3. Spatial loglikelihood
        Likespat = 0.0
        for w in 1:NW
           mu_p, sigma_p = execsacc(kpos, k, w, view, lbord, rbord, wlen, mu, sig, siglike)
           q = 1.0 / (sqrt(2 * pi) * sigma_p) * exp(-0.5 * ((mu_p - (next_kpos - 0.5)) / sigma_p)^2)
           Likespat += p[w] * q
        end
        Likespat = p[fword[j+1]]  
        LLspat += log(Likespat)

        # update fixation position
        k = fword[j+1]
        kpos = fletter[j+1]
        next_kpos = fletter[j+2]

        # 4. Temporal loglikelihood
        tfix = fdur[j+1]
        dist = Distributions.Gamma(ρ * (1.0 + ι * a[k]), scale)
        LLtime += logpdf(dist, tfix)
    end
    # return loglikelihoods
    ll = [LLtime, LLspat]
    return ll
end


function LL(corpus::DataFrame, data::DataFrame, oculo::DataFrame, δₗ, δᵣ, ν, μₜ, ι, β)
    """ Computes the lexical processing rates for each word of the sentence.
    
    # Arguments
    * `corpus` : whole corpus with columns: 'sentID', 'lfreq', 'length'
    * `data` : reading path data with column: 'sentID', 'word', 'fixpos', 'fixdur'
    * `oculo` : ocular parameters for saccade types (μ, σ)
    * `δₗ` : leftward processing span (in letters)
    * `δᵣ` : rightward processing span (in letters)
    * `ν` : shape of processing rate function
    * `μ` : ####### TODO: should be included in oculo
    * `ι` : coupling of saccade timer and activation
    * `β` : word-frequency effect

    # Returns
    * `loglik / length(SID)` : loglikelihood normalized by number of sentences

    # Notes
    * Notes can go here

    # Examples
    ```julia
    julia> 
    ```
    """
    loglik = 0.0
    SID = unique(data.sentID)
    for j in SID
        sent = subset(corpus, :sentID => ByRow(==(j)))
        lfreq = sent[!, :lfreq]
        wlen = sent[!, :length]
        NW = length(lfreq)
        pred = fill(0.0, NW)
        lbord = 0
        rbord = wlen[1] + 1
        view = wlen[1] / 2
        for w = 2:NW
            lbord = [lbord; rbord[w-1]]
            rbord = [rbord; lbord[w] + wlen[w] + 1]
            view = [view; lbord[w] + (wlen[w] + 1) / 2]
        end
        # prepare fixation sequence
        fword = subset(data, :sentID => ByRow(==(j))).word
        fletter = subset(data, :sentID => ByRow(==(j))).fixpos
        fixdur = subset(data, :sentID => ByRow(==(j))).fixdur
        # add loglik for a single sentence

        #### TODO: dont hand over μ but oculo for μ and σ (see mcmc)
        ll = LLswift2(lfreq, pred, wlen, view, lbord, rbord, fword, fletter, fixdur, δₗ, δᵣ, ν, μₜ, ι, β)
        loglik += sum(ll)
    end
    return loglik / length(SID)
end


function swiftgen(r, δₗ, δᵣ, ν, μₜ, ρ, ι, η, β, κ, γ, ω, lfreq, pred, wlen, view, lbord, rbord, mu, sig, siglike, MODE)
    """ Generates view trajectory for a sentence for visualization (MODE=1) or for modelling (MODE=2).
    
    # Arguments
    * `r` : total processing rate
    * `δₗ` : leftward processing span (in letters)
    * `δᵣ` : rightward processing span (in letters)
    * `ν` : shape of processing rate function
    * `μₜ` : mean saccade timer interval
    * `ρ` : gamma density shape parameter
    * `ι` : coupling of saccade timer and activation
    * `η` : noise in saccade targeting (log10)
    * `β` : word-frequency effect
    * `κ` : word-length exponent   ########################### TODO: OR global inhibition? (see datagen)
    * `γ` : saccade targeting exponent
    * `ω` : decay of activation
    * `lfreq` : log frequency of each word in sentence
    * `pred` : predictability of each word in sentence
    * `wlen` : length of word k
    * `view` : center positions of all words in sentence (in letters)
    * `lbord` : left border positions of all words in sentence (in letters)
    * `rbord` : right border positions of all words in sentence (in letters)
    * `mu` : vector of means for each saccade type
    * `sig` : vector of sigmas for each saccade type
    * `siglike` : standard deviation of likelihood function
    * `MODE` : visualization (1) vs. generation (2) parameter

    # Returns
    * `traj, targets` : true fixated positions, planned fixation positions

    # Notes
    * Notes can go here

    # Examples
    ```julia
    julia> 
    ```
    """
    # define dynamical variables
    NW = length(lfreq)            # number of words
    a = zeros(NW)                 # word activations
    amax = 1.5 .- β * lfreq       # maximum activations
    s = zeros(NW)                 # word saliencies
    p = zeros(NW)                 # selection probabilities
    λ = zeros(NW)                 # processing rates

    # new parameters
    scale = μₜ / ρ                 # gamma density scale parameter (mean μₜ, shape ρ)

    # other variables
    time = 0               # time
    k = 1                  # fixated word
    kpos = view[1]         # initial gaze position

    # simulation loop
    if MODE == 1
        traj = reshape(vcat(time, k, kpos, s), 1, NW + 3)      # trajectory storage w/ saliencies
    elseif MODE == 2
        traj = reshape(vcat(0.0, k, kpos), 1, 3)               # trajectory storage w/o saliencies
    end
    targets = [0 1]                                        # saccade targets storage
    while (sum(a ./ amax) < NW && k < NW)

        # 1. generate fixation duration
        dist = Distributions.Gamma(ρ, scale * (1.0 + ι * a[k]))
        tfix = rand(dist)

        # 2. compute lexical processing rates
        λ = lexrate(kpos, k, NW, view, wlen, δₗ, δᵣ, ν, κ)
        for j = 1:NW
            if a[j] > 0.5 * amax[j] && λ[j] < ω
                λ[j] = ω
            end
        end

        # 3. evolve activations and compute saliencies
        if MODE == 1
            dt = 1.0    # time step for numerical integration
            for t in collect(range(1, tfix, step=dt))
                time += dt
                a += r * λ * dt / 1000.0
                for k in 1:NW
                    a[k] = clamp(a[k], 0, amax[k])
                end
                # saliences
                s = amax .* sin.(π * a ./ amax) .+ 10.0^η

                # store trajectory
                new = reshape(vcat(time, k, kpos, s), 1, NW + 3)
                traj = [traj; new]
            end
        elseif MODE == 2
            time += tfix
            a += r * λ * tfix / 1000.0
            for k in 1:NW
                a[k] = clamp(a[k], 0, amax[k])
            end
            # saliences
            s = amax .* sin.(π * a ./ amax) .+ 10.0^η

            # store trajectory
            new = reshape(vcat(tfix, k, kpos), 1, 3)
            traj = [traj; new]
        end

        # 4. compute saliencies and selection probabilities
        s = amax .* sin.(π * a ./ amax) .+ 10.0^η
        p = (s .^ γ) ./ sum(s .^ γ)

        # 5. select saccade target
        next_tar = sample(1:NW, Weights(p))
        targets = [targets; time next_tar]

        # 6. execute saccade
        sacpar = execsacc(kpos, k, next_tar, view, lbord, rbord, wlen, mu, sig, siglike)
        k = Int64(sacpar[1])
        kpos = sacpar[2]
    end
    return traj, targets
end

end