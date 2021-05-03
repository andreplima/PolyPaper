# PolyPaper
Supplementary material from the paper "An Interpretable Recommendation Model for Gerontological Care"

***
**(1) Data used to render Figure 4 is available in filteredResults.csv.** The file uses tab as column separator. The column `essay` corresponds to a condition of the study, which is fully described by the `cutoff` and `#groups` columns. In each condition, a `model` may have been evaluated using diverse `params`. Columns `ci precision lb` and `ci precision ub` describe the confidence interval around the mean performance, and `ci precision mu` is the point estimate. Finally, `sample size` indicates the number of samples used to obtain a bootstrap estimate of the confidence interval.
It must be seen that multiple configurations of the same model may attain equivalent levels of performance. In this case, all configurations are listed. For example, in `essay` 1, several configurations of MLP attained equivalent, top level performance.

***
**(2) About the sample of patients in the original dataset.** The dataset constains 108 assessments obtained by applying the WHOQOL-BREF instrument to a sample of older individuals who are users of health services provided by a non-profit organisation.  All patients were assessed before the beginning of COVID-19 pandemic, in March, 2020.
In this sample, 93 individuals are female, aged 66.8 years on average (sd = 8.4), 14 individuals are male, aged 68.8 years on average (sd = 5.2), and one individual did not report self-identified gender.

