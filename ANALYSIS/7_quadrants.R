#!/usr/bin/env Rscript

# Model-predicted "quadrants" for the Liking x Robomentism interaction.
#
# This retains both predictors as continuous variables. "Low" and "high" are
# evaluation points at one sample SD below and above each centered mean; they
# are not used to divide participants into observed groups.

suppressPackageStartupMessages(library(MASS))

OUTCOME <- "captcha_post_completions"
LIKE_COL <- "q_post_specific_likeability"
MENT_COL <- "q_post_specific_mentism"

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) == 1) {
  script_path <- normalizePath(sub("^--file=", "", script_arg), winslash = "/")
  here <- dirname(script_path)
} else {
  here <- normalizePath(getwd(), winslash = "/")
}
out_dir <- file.path(here, "7_quadrants")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

holm_adjust <- function(p) {
  p.adjust(p, method = "holm")
}

model_cells <- function(fit, grid) {
  X <- model.matrix(delete.response(terms(fit)), grid)
  beta <- coef(fit)
  V <- vcov(fit)
  X <- X[, names(beta), drop = FALSE]
  eta <- drop(X %*% beta)
  eta_var <- diag(X %*% V %*% t(X))
  eta_se <- sqrt(pmax(eta_var, 0))

  grid$predicted_completions <- exp(eta)
  grid$response_se <- grid$predicted_completions * eta_se
  grid$asymptotic_LCL <- exp(eta - qnorm(0.975) * eta_se)
  grid$asymptotic_UCL <- exp(eta + qnorm(0.975) * eta_se)
  list(cells = grid, X = X, beta = beta, V = V, eta = eta)
}

pairwise_contrasts <- function(cell_result) {
  cells <- cell_result$cells
  pairs <- combn(seq_len(nrow(cells)), 2)
  rows <- lapply(seq_len(ncol(pairs)), function(k) {
    i <- pairs[1, k]
    j <- pairs[2, k]
    cvec <- cell_result$X[i, ] - cell_result$X[j, ]
    estimate <- drop(cvec %*% cell_result$beta)
    se <- sqrt(drop(cvec %*% cell_result$V %*% cvec))
    z <- estimate / se
    p <- 2 * pnorm(abs(z), lower.tail = FALSE)
    data.frame(
      contrast = paste(cells$cell[i], "-", cells$cell[j]),
      numerator = cells$cell[i],
      denominator = cells$cell[j],
      log_rate_ratio = estimate,
      SE = se,
      z_ratio = z,
      p_value = p,
      rate_ratio = exp(estimate),
      asymptotic_LCL = exp(estimate - qnorm(0.975) * se),
      asymptotic_UCL = exp(estimate + qnorm(0.975) * se),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  out$p_value_holm <- holm_adjust(out$p_value)
  out$holm_significant_05 <- out$p_value_holm < 0.05
  out
}

plot_predicted_cells <- function(cells, path, dataset_label) {
  cols <- c("Low Robomentism (-1 SD)" = "#2F5D8A", "High Robomentism (+1 SD)" = "#B24A2A")
  pch_values <- c("Low Robomentism (-1 SD)" = 16, "High Robomentism (+1 SD)" = 17)
  ylim <- range(c(0, cells$asymptotic_LCL, cells$asymptotic_UCL), finite = TRUE)

  png(path, width = 1800, height = 1250, res = 220)
  par(mar = c(5, 5, 4, 2) + 0.1)
  plot(
    NA, xlim = c(0.8, 2.2), ylim = ylim, xaxt = "n",
    xlab = "Liking evaluation point", ylab = "Predicted Voluntary reCAPTCHA Solutions",
    bty = "l"
  )
  axis(1, at = 1:2, labels = FALSE)
  mtext(
    c("Low Liking\n(-1 SD)", "High Liking\n(+1 SD)"),
    side = 1, at = 1:2, line = 2
  )
  grid(nx = NA, ny = NULL, col = "#E5E5E5")

  for (level in names(cols)) {
    d <- cells[cells$Robomentism_level == level, ]
    x <- match(d$Liking_level, c("Low Liking (-1 SD)", "High Liking (+1 SD)"))
    lines(x, d$predicted_completions, col = cols[level], lwd = 2)
    arrows(
      x, d$asymptotic_LCL, x, d$asymptotic_UCL,
      angle = 90, code = 3, length = 0.06, col = cols[level], lwd = 1.6
    )
    points(
      x, d$predicted_completions, col = cols[level],
      pch = pch_values[level], cex = 1.35
    )
  }
  legend(
    "bottomright", legend = names(cols), col = cols, pch = pch_values,
    lty = 1, lwd = 2, bty = "n"
  )

  dev.off()
}

plot_contrasts <- function(contrasts, path, dataset_label) {
  d <- contrasts[order(contrasts$rate_ratio), ]
  y <- seq_len(nrow(d))
  point_cols <- ifelse(d$holm_significant_05, "#B24A2A", "#2F5D8A")
  short_cell <- function(x) {
    Liking <- ifelse(grepl("^Low Liking", x), "L", "H")
    Robomentism <- ifelse(grepl(" / Low Robomentism", x), "L", "H")
    paste0(Liking, Robomentism)
  }
  short_labels <- paste(
    short_cell(d$numerator), "-", short_cell(d$denominator)
  )

  png(path, width = 2200, height = 1450, res = 220)
  par(mar = c(6, 5, 4, 2) + 0.1)
  plot(
    d$rate_ratio, y, log = "x",
    xlim = range(c(1, d$asymptotic_LCL, d$asymptotic_UCL), finite = TRUE),
    yaxt = "n", ylab = "", xlab = "Rate ratio (numerator / denominator; log scale)",
    main = paste("Pairwise model-predicted cell contrasts:", dataset_label),
    pch = 16, col = point_cols, bty = "l"
  )
  axis(2, at = y, labels = short_labels, las = 1)
  abline(v = 1, lty = 2, col = "#555555")
  segments(d$asymptotic_LCL, y, d$asymptotic_UCL, y, col = point_cols, lwd = 1.6)
  points(d$rate_ratio, y, pch = 16, col = point_cols)
  legend(
    "bottomright", legend = c("Holm p < .05", "Holm p >= .05"),
    col = c("#B24A2A", "#2F5D8A"), pch = 16, bty = "n"
  )
  mtext(
    "LL = low Liking/low Robomentism; HL = high/low; LH = low/high; HH = high/high.",
    side = 1, line = 4, cex = 0.78
  )
  mtext(
    "Intervals are unadjusted 95% CIs; colour uses Holm-adjusted p-values.",
    side = 1, line = 5, cex = 0.78
  )
  dev.off()
}

run_quadrants <- function(csv_path, dataset_label) {
  raw <- read.csv(csv_path, check.names = FALSE)
  required <- c(OUTCOME, LIKE_COL, MENT_COL)
  missing <- setdiff(required, names(raw))
  if (length(missing) > 0) {
    stop("Missing required columns: ", paste(missing, collapse = ", "))
  }

  dat <- raw[, required]
  dat[] <- lapply(dat, function(x) suppressWarnings(as.numeric(x)))
  dat <- dat[complete.cases(dat), ]
  if (nrow(dat) == 0) stop("No complete rows remain for ", dataset_label)

  like_mean <- mean(dat[[LIKE_COL]])
  ment_mean <- mean(dat[[MENT_COL]])
  dat$like_c <- dat[[LIKE_COL]] - like_mean
  dat$Robomentism_c <- dat[[MENT_COL]] - ment_mean
  like_sd <- sd(dat$like_c)
  Robomentism_sd <- sd(dat$Robomentism_c)
  if (!is.finite(like_sd) || like_sd == 0 || !is.finite(Robomentism_sd) || Robomentism_sd == 0) {
    stop("Liking and Robomentism must both have non-zero finite SDs.")
  }

  fit <- glm.nb(
    captcha_post_completions ~ like_c * Robomentism_c,
    data = dat
  )

  grid <- expand.grid(
    like_c = c(-like_sd, like_sd),
    Robomentism_c = c(-Robomentism_sd, Robomentism_sd),
    KEEP.OUT.ATTRS = FALSE
  )
  grid$Liking_level <- ifelse(grid$like_c < 0, "Low Liking (-1 SD)", "High Liking (+1 SD)")
  grid$Robomentism_level <- ifelse(grid$Robomentism_c < 0, "Low Robomentism (-1 SD)", "High Robomentism (+1 SD)")
  grid$cell <- paste(grid$Liking_level, grid$Robomentism_level, sep = " / ")
  grid$Liking_raw_value <- grid$like_c + like_mean
  grid$Robomentism_raw_value <- grid$Robomentism_c + ment_mean

  cell_result <- model_cells(fit, grid)
  cells <- cell_result$cells
  contrasts <- pairwise_contrasts(cell_result)

  prefix <- file.path(out_dir, paste0("7_quadrants_", dataset_label, "_"))
  cells_path <- paste0(prefix, "predicted_cells_", stamp, ".csv")
  contrasts_path <- paste0(prefix, "pairwise_contrasts_holm_", stamp, ".csv")
  cell_plot_path <- paste0(prefix, "predicted_cells_", stamp, ".png")
  contrast_plot_path <- paste0(prefix, "pairwise_contrasts_", stamp, ".png")
  summary_path <- paste0(prefix, "summary_", stamp, ".txt")

  write.csv(cells, cells_path, row.names = FALSE)
  write.csv(contrasts, contrasts_path, row.names = FALSE)
  plot_predicted_cells(cells, cell_plot_path, dataset_label)
  plot_contrasts(contrasts, contrast_plot_path, dataset_label)

  capture.output(
    cat(
      "Model-predicted Liking x Robomentism cells\n",
      "dataset: ", dataset_label, "\n",
      "data: ", normalizePath(csv_path, winslash = "/"), "\n",
      "rows used: ", nrow(dat), "\n",
      "model: captcha_post_completions ~ like_c * Robomentism_c\n",
      "negative-binomial parameterization: MASS::glm.nb (log link)\n",
      "Liking mean: ", like_mean, "\n",
      "Liking SD: ", like_sd, "\n",
      "Robomentism mean: ", ment_mean, "\n",
      "Robomentism SD: ", Robomentism_sd, "\n",
      "theta: ", fit$theta, "\n\n",
      sep = ""
    ),
    cat("Model coefficients\n"),
    print(coef(summary(fit))),
    cat("\nPredicted response-scale cells\n"),
    print(cells),
    cat("\nAll pairwise rate-ratio contrasts (Holm-adjusted p-values)\n"),
    print(contrasts),
    file = summary_path
  )

  cat("\nSaved quadrant analysis for '", dataset_label, "':\n", sep = "")
  cat("- ", cells_path, "\n", sep = "")
  cat("- ", contrasts_path, "\n", sep = "")
  cat("- ", cell_plot_path, "\n", sep = "")
  cat("- ", contrast_plot_path, "\n", sep = "")
  cat("- ", summary_path, "\n", sep = "")
}

primary_csv <- Sys.getenv(
  "ANALYSIS_INPUT_CSV",
  unset = file.path(here, "3_purified.csv")
)
run_quadrants(primary_csv, "primary")

sensitivity_csv <- Sys.getenv(
  "ANALYSIS_SENSITIVITY_INPUT_CSV",
  unset = file.path(here, "3_dfbetas_sensitivity.csv")
)
if (file.exists(sensitivity_csv)) {
  run_quadrants(sensitivity_csv, "dfbetas_sensitivity")
} else {
  cat("\nNo DFBETAS sensitivity dataset found at: ", sensitivity_csv, "\n", sep = "")
}
