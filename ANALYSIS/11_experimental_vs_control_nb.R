#!/usr/bin/env Rscript

# Experimental/competition versus no-competition control comparison.
#
# The purified dataset contains the project's established exclusions. The
# original experimental recruitment used participant numbers 1--103. New
# control participants are identified only by participant numbers 200--299.
# Numbers outside those two intended ranges are excluded from this analysis.

suppressPackageStartupMessages(library(MASS))

OUTCOME <- "captcha_post_completions"
PARTICIPANT_ID <- "participant_number"
EXPERIMENTAL_MIN <- 1L
EXPERIMENTAL_MAX <- 103L
CONTROL_MIN <- 200L
CONTROL_MAX <- 299L

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) == 1) {
  script_path <- normalizePath(sub("^--file=", "", script_arg), winslash = "/")
  here <- dirname(script_path)
} else {
  here <- normalizePath(getwd(), winslash = "/")
}

input_csv <- Sys.getenv(
  "ANALYSIS_INPUT_CSV",
  unset = file.path(here, "3_purified.csv")
)
out_dir <- Sys.getenv(
  "ANALYSIS_CONTROL_OUTPUT_DIR",
  unset = file.path(here, "11_control")
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

fmt <- function(x, digits = 3) {
  formatC(x, digits = digits, format = "f")
}

fmt_p <- function(p) {
  if (!is.finite(p)) return("NA")
  if (p < 0.001) return("< .001")
  sub("^0", "", formatC(p, digits = 3, format = "f"))
}

condition_descriptives <- function(dat) {
  rows <- lapply(levels(dat$condition), function(group) {
    x <- dat[[OUTCOME]][dat$condition == group]
    zeros <- sum(x == 0)
    data.frame(
      condition = group,
      N = length(x),
      mean = mean(x),
      SD = if (length(x) > 1) sd(x) else NA_real_,
      median = median(x),
      minimum = min(x),
      maximum = max(x),
      zero_N = zeros,
      zero_percent = 100 * zeros / length(x),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

expected_counts <- function(fit) {
  newdata <- data.frame(
    condition = factor(
      c("control", "experimental"),
      levels = c("control", "experimental")
    )
  )
  X <- model.matrix(delete.response(terms(fit)), newdata)
  beta <- coef(fit)
  X <- X[, names(beta), drop = FALSE]
  eta <- drop(X %*% beta)
  eta_se <- sqrt(pmax(diag(X %*% vcov(fit) %*% t(X)), 0))
  data.frame(
    condition = as.character(newdata$condition),
    expected_count = exp(eta),
    CI_low = exp(eta - qnorm(0.975) * eta_se),
    CI_high = exp(eta + qnorm(0.975) * eta_se),
    stringsAsFactors = FALSE
  )
}

plot_observed <- function(dat, path) {
  group_number <- as.integer(dat$condition)
  set.seed(20260912)
  x <- jitter(group_number, amount = 0.10)
  y <- log1p(dat[[OUTCOME]])
  group_cols <- c(control = "#2F5D8A", experimental = "#B24A2A")

  png(path, width = 1700, height = 1200, res = 220)
  par(mar = c(5, 5, 3, 2) + 0.1)
  plot(
    x, y,
    xlim = c(0.65, 2.35), xaxt = "n", yaxt = "n",
    xlab = "Condition", ylab = "Voluntary reCAPTCHA completions",
    pch = 16, cex = 0.85,
    col = adjustcolor(group_cols[as.character(dat$condition)], alpha.f = 0.58),
    bty = "l"
  )
  axis(1, at = 1:2, labels = c("No-competition control", "Experimental/competition"))
  tick_values <- unique(pretty(c(0, dat[[OUTCOME]])))
  tick_values <- tick_values[tick_values >= 0]
  axis(2, at = log1p(tick_values), labels = tick_values, las = 1)
  grid(nx = NA, ny = NULL, col = "#E5E5E5")

  for (i in 1:2) {
    values <- dat[[OUTCOME]][group_number == i]
    quartiles <- quantile(values, probs = c(0.25, 0.50, 0.75), names = FALSE)
    segments(i, log1p(quartiles[1]), i, log1p(quartiles[3]), lwd = 6,
             col = group_cols[i])
    segments(i - 0.11, log1p(quartiles[2]), i + 0.11, log1p(quartiles[2]),
             lwd = 3, col = "white")
    points(i, log1p(mean(values)), pch = 23, cex = 1.25,
           bg = "#F2C14E", col = "#333333")
  }
  legend(
    "topright",
    legend = c("Individual observation", "Median and IQR", "Arithmetic mean"),
    pch = c(16, NA, 23), lty = c(NA, 1, NA), lwd = c(NA, 5, NA),
    col = c("#777777", "#555555", "#333333"),
    pt.bg = c(NA, NA, "#F2C14E"), bty = "n"
  )
  mtext("Display uses a log(1 + count) axis so zeros and the right tail remain visible.",
        side = 1, line = 4, cex = 0.75)
  dev.off()
}

raw <- read.csv(input_csv, check.names = FALSE, colClasses = "character")
required <- c(PARTICIPANT_ID, OUTCOME)
missing_columns <- setdiff(required, names(raw))
if (length(missing_columns) > 0) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}

# Parse participant numbers strictly: blanks are missing, and all other values
# must be non-negative integer representations (for example, 200 or 200.0).
participant_text <- trimws(raw[[PARTICIPANT_ID]])
participant_missing <- is.na(participant_text) | participant_text == ""
participant_format_ok <- participant_missing | grepl("^[0-9]+([.]0+)?$", participant_text)
participant_numeric <- suppressWarnings(as.numeric(participant_text))
participant_integer_ok <- participant_missing |
  (is.finite(participant_numeric) & participant_numeric == floor(participant_numeric) &
     participant_numeric <= .Machine$integer.max)
participant_parse_ok <- participant_format_ok & participant_integer_ok

if (any(!participant_parse_ok)) {
  bad <- unique(participant_text[!participant_parse_ok])
  stop("Participant-number parsing failed for: ", paste(bad, collapse = ", "))
}
raw$participant_number_parsed <- as.integer(participant_numeric)

is_experimental <- !participant_missing &
  raw$participant_number_parsed >= EXPERIMENTAL_MIN &
  raw$participant_number_parsed <= EXPERIMENTAL_MAX
is_control <- !participant_missing &
  raw$participant_number_parsed >= CONTROL_MIN &
  raw$participant_number_parsed <= CONTROL_MAX
ambiguous <- is_experimental & is_control
if (any(ambiguous)) stop("At least one participant was assigned ambiguously.")

in_intended_sample <- xor(is_experimental, is_control)
outside_numbers <- sort(unique(raw$participant_number_parsed[!in_intended_sample & !participant_missing]))
missing_number_N <- sum(participant_missing)

dat <- raw[in_intended_sample, c(PARTICIPANT_ID, OUTCOME)]
dat$participant_number <- raw$participant_number_parsed[in_intended_sample]
dat$condition <- ifelse(is_control[in_intended_sample], "control", "experimental")

duplicate_numbers <- sort(unique(dat$participant_number[duplicated(dat$participant_number)]))
if (length(duplicate_numbers) > 0) {
  stop("Duplicate participant numbers in the intended sample: ",
       paste(duplicate_numbers, collapse = ", "))
}

outcome_text <- trimws(dat[[OUTCOME]])
outcome_missing <- is.na(outcome_text) | outcome_text == ""
outcome_numeric <- suppressWarnings(as.numeric(outcome_text))
outcome_valid <- outcome_missing |
  (is.finite(outcome_numeric) & outcome_numeric >= 0 &
     outcome_numeric == floor(outcome_numeric))
if (any(!outcome_valid)) {
  bad_ids <- dat$participant_number[!outcome_valid]
  stop("Non-negative integer count check failed for participant(s): ",
       paste(bad_ids, collapse = ", "))
}

missing_outcome_N <- sum(outcome_missing)
dat <- dat[!outcome_missing, ]
dat[[OUTCOME]] <- as.integer(outcome_numeric[!outcome_missing])
dat$condition <- factor(dat$condition, levels = c("control", "experimental"))

control_range_check <- all(
  dat$participant_number[dat$condition == "control"] >= CONTROL_MIN &
    dat$participant_number[dat$condition == "control"] <= CONTROL_MAX
)
if (!control_range_check) stop("Control range check failed.")
if (anyNA(dat$condition)) stop("At least one included participant has no condition assignment.")

group_N <- table(dat$condition)
cat("Participant-number checks\n")
cat("- Rows read: ", nrow(raw), "\n", sep = "")
cat("- Successfully parsed non-missing participant numbers: ",
    sum(!participant_missing), "\n", sep = "")
cat("- Missing participant numbers excluded: ", missing_number_N, "\n", sep = "")
cat("- Ambiguous assignments: ", sum(ambiguous), "\n", sep = "")
cat("- Experimental range: ", EXPERIMENTAL_MIN, "-", EXPERIMENTAL_MAX, "\n", sep = "")
cat("- Control range: ", CONTROL_MIN, "-", CONTROL_MAX, "\n", sep = "")
cat("- Control range check: PASSED\n")
cat("- Outside-range participant numbers excluded: ",
    if (length(outside_numbers) == 0) "none" else paste(outside_numbers, collapse = ", "),
    "\n", sep = "")
cat("- Missing outcomes excluded: ", missing_outcome_N, "\n", sep = "")
cat("- Non-negative integer outcome check: PASSED\n")
cat("- Experimental N in model: ", unname(group_N["experimental"]), "\n", sep = "")
cat("- Control N in model: ", unname(group_N["control"]), "\n\n", sep = "")

# A condition must contain at least one observation for a two-group contrast;
# beyond that mathematical requirement, no minimum group size is imposed.
if (any(group_N == 0)) {
  stop("Both conditions must contain at least one valid observation to fit the comparison.")
}

descriptives <- condition_descriptives(dat)
model_formula <- captcha_post_completions ~ condition
fit <- glm.nb(model_formula, data = dat)
coefficient_matrix <- coef(summary(fit))
coefficient_table <- data.frame(
  term = rownames(coefficient_matrix),
  estimate = coefficient_matrix[, "Estimate"],
  standard_error = coefficient_matrix[, "Std. Error"],
  z = coefficient_matrix[, "z value"],
  p_value = coefficient_matrix[, "Pr(>|z|)"],
  row.names = NULL,
  check.names = FALSE
)

condition_term <- "conditionexperimental"
if (!condition_term %in% coefficient_table$term) {
  stop("Expected experimental-versus-control coefficient was not found.")
}
condition_row <- coefficient_table[coefficient_table$term == condition_term, ]
log_irr <- condition_row$estimate
log_irr_se <- condition_row$standard_error
irr <- exp(log_irr)
irr_ci <- exp(log_irr + c(-1, 1) * qnorm(0.975) * log_irr_se)
predictions <- expected_counts(fit)

control_mean <- descriptives$mean[descriptives$condition == "control"]
experimental_mean <- descriptives$mean[descriptives$condition == "experimental"]
raw_mean_difference <- experimental_mean - control_mean

if (irr_ci[1] > 1) {
  interpretation <- paste0(
    "The experimental condition was associated with a higher estimated rate of ",
    "voluntary reCAPTCHA completion than the no-competition control condition; ",
    "the 95% CI for the IRR excluded 1."
  )
} else if (irr_ci[2] < 1) {
  interpretation <- paste0(
    "The experimental condition was associated with a lower estimated rate of ",
    "voluntary reCAPTCHA completion than the no-competition control condition; ",
    "the 95% CI for the IRR excluded 1."
  )
} else if (irr > 1) {
  interpretation <- paste0(
    "The estimated completion rate was higher in the experimental condition, ",
    "but the 95% CI for the IRR included 1."
  )
} else if (irr < 1) {
  interpretation <- paste0(
    "The estimated completion rate was lower in the experimental condition, ",
    "but the 95% CI for the IRR included 1."
  )
} else {
  interpretation <- "The estimated completion rates were equal in the two conditions."
}

summary_line <- paste0(
  "Experimental N = ", unname(group_N["experimental"]),
  "; control N = ", unname(group_N["control"]),
  ". Observed mean completions were ", fmt(experimental_mean, 2),
  " in the experimental condition and ", fmt(control_mean, 2),
  " in the control condition. The experimental-versus-control IRR was ",
  fmt(irr, 3), " (95% CI ", fmt(irr_ci[1], 3), " to ",
  fmt(irr_ci[2], 3), "), p ", fmt_p(condition_row$p_value),
  ", theta = ", fmt(fit$theta, 3), ". ", interpretation
)

prefix <- file.path(out_dir, paste0("11_experimental_vs_control_nb_"))
descriptives_path <- paste0(prefix, "descriptives_", stamp, ".csv")
coefficients_path <- paste0(prefix, "coefficients_", stamp, ".csv")
predictions_path <- paste0(prefix, "expected_counts_", stamp, ".csv")
plot_path <- paste0(prefix, "observed_distribution_", stamp, ".png")
summary_path <- paste0(prefix, "summary_", stamp, ".txt")

write.csv(descriptives, descriptives_path, row.names = FALSE)
write.csv(coefficient_table, coefficients_path, row.names = FALSE)
write.csv(predictions, predictions_path, row.names = FALSE)
plot_observed(dat, plot_path)

report <- capture.output({
  cat("Experimental/competition versus no-competition control analysis\n")
  cat("Data: ", normalizePath(input_csv, winslash = "/"), "\n", sep = "")
  cat("Model formula: ", paste(deparse(formula(fit)), collapse = " "), "\n", sep = "")
  cat("Negative-binomial model: MASS::glm.nb with log link\n")
  cat("Reference category: control\n\n")

  cat("Condition descriptives\n")
  print(descriptives, row.names = FALSE)
  cat("\nModel coefficients\n")
  print(coefficient_table, row.names = FALSE)
  cat("\nEstimated theta/dispersion: ", fit$theta,
      " (SE = ", fit$SE.theta, ")\n", sep = "")
  cat("Experimental versus control IRR: ", irr,
      " (95% CI ", irr_ci[1], " to ", irr_ci[2], ")\n", sep = "")
  cat("Raw arithmetic mean difference (experimental - control): ",
      raw_mean_difference, " completions\n", sep = "")
  cat("\nModel-estimated expected completion counts\n")
  print(predictions, row.names = FALSE)
})

writeLines(c(report, "", "Plain-language summary", summary_line), summary_path)
cat(paste(report, collapse = "\n"), "\n")
cat("\nSaved control-comparison outputs:\n")
cat("- ", descriptives_path, "\n", sep = "")
cat("- ", coefficients_path, "\n", sep = "")
cat("- ", predictions_path, "\n", sep = "")
cat("- ", plot_path, "\n", sep = "")
cat("- ", summary_path, "\n", sep = "")
cat("\nPlain-language summary\n")
cat(summary_line, "\n")
