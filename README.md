# MATH60633A – TP2: Options Risk Management

Assignment 2 for MATH60633A (Statistical Methods) at HEC Montréal.

## Overview

This project prices a portfolio of S&P 500 options and estimates one-week risk measures (VaR and ES) under several modeling approaches of increasing complexity.

## Sections

| # | Topic | Points |
|---|-------|--------|
| 0 | Data loading & exploration | — |
| 1 | Portfolio pricing (Black-Scholes) | 5 |
| 2 | One risk driver — Univariate Gaussian model | 10 |
| 3 | Two risk drivers — Bivariate Gaussian model | 15 |
| 4 | Two risk drivers — Copula-marginal model (Student-t margins) | 20 |
| 5 | Volatility surface fitting & repricing | 20 |
| 6 | Summary & comparison of risk measures | — |

## Repository Structure

```
.
├── report.Rmd              # Source R Markdown (full analysis)
├── report.pdf              # Compiled PDF report
├── R/functions.R           # Helper functions (BS pricing, copula, surface)
├── Market.rda              # Market data (S&P 500, VIX, rates, options)
└── MATH-60633A-tp-02.pdf   # Assignment specification
```

## Requirements

- R (≥ 4.0)
- Packages: `xts`, `MASS`

## Reproducing the Report

Open `Assignment-2.Rproj` in RStudio, then knit `report.Rmd`:

```r
rmarkdown::render("report.Rmd")
```
