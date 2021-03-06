# PolyPaper
Supplementary material from the paper "An Interpretable Recommendation Model for Gerontological Care"


---

In the `prototype` folder, you will find three scripts:

*   `simulate.py`: builds a synthetic dataset of patient assessments (similar to the dataset described in Section 4 of the paper), performs the learning process detailed in Section 3.3 to learn representations for interventions (i.e., treatments), and evaluates these representations.
A dendrogram similar to the one shown in Figure 3 is created in the current folder, as well as CSV-file detailing how each sample in the dataset contributed to the overall performance.
This script accepts two optional command-line parameters: sample size and cutoff level.
Example: `python simulate.py 30 10`.
(This script takes about 10min to complete in a 4-core computer.)

*   `plotGrid.py`: plots two grids showing how each patient may potentially benefit from each intervention.
The plots are saved in the `patient_treament_train.pdf` and `patient_treament_test.pdf` files.
Example: `python plotGrid.py`.

*   `animateGrid.py`: builds an animation that retraces the learning process from which the representations for interventions were obtained. The animation is
saved in the `retrace_learning.mp4` file. Example: `python animateGrid.py`.

The requirements of the environment needed to run the scripts in this prototype is specified in the `polypaper.yml` file.
To create an environment using conda, use `conda env create -f polypaper.yml`.

In the `examples` folder, you will find files produced by these scripts.

---


### Additional notes:

**(1) The data used to render Figure 4 is available in `data/filteredResults.csv` file.** 
The file uses TAB as column separator. 
The column `essay` corresponds to a condition of the study, which is fully described by the `cutoff` and `#groups` columns. 
In each condition, a `model` may have been evaluated using diverse `params`.
Columns `ci precision lb` and `ci precision ub` describe the confidence interval around the mean performance, and `ci precision mu` is the point estimate.
Finally, `sample size` indicates the number of samples used to obtain a bootstrap estimate of the confidence interval.
It must be seen that multiple configurations of the same model may attain equivalent levels of performance.
In this case, all configurations are listed. For example, in `essay` 1, several configurations of MLP attained equivalent, top level performance.

**(2) About the sample of patients in the original dataset.**
The dataset contains 108 assessments obtained by applying the WHOQOL-BREF instrument to a sample of older individuals who are users of health services provided by a local, non-profit organisation.
All patients were assessed before the beginning of COVID-19 pandemic, in March, 2020.
In this sample, 93 individuals are female, aged 66.8 years on average (sd = 8.4), 14 individuals are male, aged 68.8 years on average (sd = 5.2), and one individual did not report self-identified gender.
This study in which these data were collected was approved by the S??o Carlos Federal University Ethics Committee (process 1.104.750/2015).
