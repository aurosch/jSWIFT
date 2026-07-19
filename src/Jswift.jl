module Jswift

"""
module structure:

export plustwo

plustwo(x) = return x+2
"""

#export LLswift2
export execsacc
#export lexrate
#export LL

function execsacc(kpos, k, next_tar, view, lbord, rbord, wlen, mu, sig, siglike)
    """ This is the execsacc function.

    # Arguments
    * `kpos` : current absolute letter position
    * `k` : current word (int)
    * `next_tar` : word chosen for next target (int)
    * `view` : absolute position of center of k in sentence
    * `lbord` : absolute position of left border of word in sentence
    * `rbord` : absolute position of right border of word in sentence
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

end