# =============================================================================
# Helper Functions for Options Risk Management
# MATH60633A - TP2
# =============================================================================

#' Black-Scholes European Call Option Price
#'
#' @param S Spot price (scalar or vector)
#' @param K Strike price (scalar or vector)
#' @param r Risk-free rate (scalar or vector)
#' @param sigma Implied volatility (scalar or vector)
#' @param tau Time to maturity in years (scalar or vector)
#' @return Call option price(s)
bs_call <- function(S, K, r, sigma, tau) {
  d1 <- (log(S / K) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
  d2 <- d1 - sigma * sqrt(tau)
  S * pnorm(d1) - K * exp(-r * tau) * pnorm(d2)
}

#' Linear Interpolation of the Term Structure
#'
#' @param rf_mats Numeric vector of maturities (in years)
#' @param rf_vals Numeric vector of corresponding rates
#' @param tau_rf  Target maturity in years (using 360-day convention)
#' @return Interpolated rate
interpolate_rate <- function(rf_mats, rf_vals, tau_rf) {
  approx(x = rf_mats, y = rf_vals, xout = tau_rf, rule = 2)$y
}

#' Price the 4-Option Book
#'
#' Book: 1x K=1600 T=days1, 1x K=1650 T=days1,
#'       1x K=1750 T=days2, 1x K=1800 T=days2
#'
#' @param S       Spot price(s) — scalar or vector of length n_sim
#' @param sigma   Implied volatility — scalar or vector of length n_sim
#' @param rf_mats Numeric vector of term structure maturities
#' @param rf_vals Numeric vector of term structure rates
#' @param days1   Days to maturity for the first two options
#' @param days2   Days to maturity for the last two options
#' @return Portfolio value(s)
portfolio_value <- function(S, sigma, rf_mats, rf_vals, days1, days2) {
  # Option time to maturity: 250 days/year
  tau_opt1 <- days1 / 250
  tau_opt2 <- days2 / 250

  # Term structure time: 360 days/year
  tau_rf1 <- days1 / 360
  tau_rf2 <- days2 / 360

  # Interpolate risk-free rates
  r1 <- interpolate_rate(rf_mats, rf_vals, tau_rf1)
  r2 <- interpolate_rate(rf_mats, rf_vals, tau_rf2)

  # Price each option (vectorized over S and sigma)
  c1 <- bs_call(S, 1600, r1, sigma, tau_opt1)
  c2 <- bs_call(S, 1650, r1, sigma, tau_opt1)
  c3 <- bs_call(S, 1750, r2, sigma, tau_opt2)
  c4 <- bs_call(S, 1800, r2, sigma, tau_opt2)

  c1 + c2 + c3 + c4
}

#' Parametric Volatility Surface
#'
#' sigma(m, tau) = alpha1 + alpha2*(m-1)^2 + alpha3*(m-1)^3 + alpha4*sqrt(tau)
#'
#' @param m     Moneyness K/S (scalar or vector)
#' @param tau   Time to maturity in years (scalar or vector)
#' @param alpha Numeric vector of 4 parameters
#' @return Model implied volatility
sigma_hat <- function(m, tau, alpha) {
  alpha[1] + alpha[2] * (m - 1)^2 + alpha[3] * (m - 1)^3 + alpha[4] * sqrt(tau)
}

#' Adjusted Volatility Surface (anchored to VIX)
#'
#' sigma_tilde(m, tau) = sigma_hat(m, tau) - (alpha1 + alpha4) + VIX
#'
#' @param m           Moneyness K/S
#' @param tau         Time to maturity in years
#' @param alpha       Fitted surface parameters
#' @param vix_current Current VIX level
#' @return Adjusted implied volatility (floored at 0.001)
sigma_tilde <- function(m, tau, alpha, vix_current) {
  vol <- sigma_hat(m, tau, alpha) - (alpha[1] + alpha[4]) + vix_current
  pmax(vol, 0.001)  # floor to avoid negative vol
}

#' Price the 4-Option Book Using the Volatility Surface
#'
#' @param S           Spot price(s)
#' @param vix_current VIX level(s) for anchoring the surface
#' @param rf_mats     Term structure maturities
#' @param rf_vals     Term structure rates
#' @param days1       Days to maturity for first two options
#' @param days2       Days to maturity for last two options
#' @param alpha       Fitted surface parameters (length 4)
#' @return Portfolio value(s)
portfolio_value_surface <- function(S, vix_current, rf_mats, rf_vals,
                                    days1, days2, alpha) {
  K_vec   <- c(1600, 1650, 1750, 1800)
  days    <- c(days1, days1, days2, days2)
  tau_opt <- days / 250
  tau_rf  <- days / 360

  # Interpolate rates (same for all scenarios)
  r_vec <- sapply(tau_rf, function(t) interpolate_rate(rf_mats, rf_vals, t))

  # Price each option with its own vol from the surface
  total <- rep(0, length(S))
  for (j in 1:4) {
    m_j   <- K_vec[j] / S
    vol_j <- sigma_tilde(m_j, tau_opt[j], alpha, vix_current)
    total <- total + bs_call(S, K_vec[j], r_vec[j], vol_j, tau_opt[j])
  }
  total
}

#' Generate Samples from a Gaussian Copula (manual implementation)
#'
#' Generates n bivariate uniform samples with Gaussian copula dependence.
#' Uses Cholesky decomposition of the correlation matrix.
#'
#' @param n   Number of samples
#' @param rho Copula correlation parameter
#' @return n x 2 matrix of uniform [0,1] samples
rgauss_copula <- function(n, rho) {
  # Cholesky factor of correlation matrix [[1, rho], [rho, 1]]
  L <- matrix(c(1, 0, rho, sqrt(1 - rho^2)), nrow = 2)
  Z <- matrix(rnorm(n * 2), nrow = n, ncol = 2) %*% t(L)
  pnorm(Z)  # transform to uniform margins
}

#' Estimate Gaussian Copula Correlation from Data
#'
#' Converts data to pseudo-observations (ranks), computes Kendall's tau,
#' and inverts the relationship tau = (2/pi) * arcsin(rho) to get rho.
#'
#' @param x Numeric vector (first variable)
#' @param y Numeric vector (second variable)
#' @return Estimated copula correlation parameter
fit_gauss_copula <- function(x, y) {
  tau_hat <- cor(x, y, method = "kendall")
  sin(tau_hat * pi / 2)  # inversion formula for Gaussian copula
}
