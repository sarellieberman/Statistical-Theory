#!/usr/bin/env python3
"""
analysis_script.py
Statistical Theory - Final Project : Student Alcohol Consumption

Run hypothesis tests and visualisations for up to six predefined
research questions (Q1-Q6).  By default, runs them all.

Author(s): <Your Names>
Date:     2025-07-07
Python:   3.11
"""

# --------------------------------------------------------------------- #
# 1. Imports and config
# --------------------------------------------------------------------- #
import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# --------------------------------------------------------------------- #
# 2. Helper functions
# --------------------------------------------------------------------- #
def load_and_concat(mat_path: Path, por_path: Path) -> pd.DataFrame:
    """
    Load the Math and Portuguese CSVs (semicolon-delimited),
    concatenate them, add a 'subject' column,
    and standardise column names to lower-case with no whitespace.
    """
    mat = pd.read_csv(mat_path, sep=';')
    por = pd.read_csv(por_path, sep=';')

    mat['subject'] = 'math'
    por['subject'] = 'portuguese'

    df = pd.concat([mat, por], axis=0)

    print(f"[INFO] Loaded {len(df)} rows with {df.shape[1]} columns")
    return df


def descriptive_stats(df: pd.DataFrame, summary_file):
    """Print basic descriptives of grades and main predictors."""
    print("\n==== Descriptive statistics ====\n")
    num_cols = [
        'G1', 'G2', 'G3', 'Dalc', 'Walc',
        'absences', 'studytime', 'health'
    ]
    desc = df[num_cols]
    print(desc, "\n")
    summary_file.write("Descriptive statistics:\n")
    summary_file.write(desc.to_string())
    summary_file.write("\n\n")


def save_figure(fig, filename: Path):
    """Save and close a matplotlib figure."""
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"[FIG] Saved → {filename}")


# --------------------------------------------------------------------- #
# 3. Research-question analyses
# --------------------------------------------------------------------- #
def question_1_alcohol(df, fout):
    """Q1 - Alcohol consumption vs grades."""
    fout.write("Q1: Alcohol consumption vs grades\n")
    r, p = stats.pearsonr(df['Walc'], df['G3'])
    fout.write(f" Pearson r (Walc vs G3) = {r:.3f},  p = {p:.4g}\n")

    model = smf.ols('G3 ~ C(Walc)', data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    fout.write(" ANOVA on G3 ~ C(Walc):\n")
    fout.write(anova.to_string()); fout.write("\n\n")

    fig, ax = plt.subplots(figsize=(4.2, 3))
    sns.boxplot(data=df, x='Walc', y='G3', ax=ax, palette='Blues')
    ax.set_xlabel("Weekend alcohol use (1 = none … 5 = heavy)")
    ax.set_ylabel("Final grade (G3)")
    ax.set_title("Grades by weekend alcohol use")
    save_figure(fig, Path("figures/fig2_grades_by_alcohol.png"))


def question_2_absences(df, fout):
    """Q2 - absences vs grades."""
    fout.write("Q2: absences vs grades\n")
    r, p = stats.pearsonr(df['absences'], df['G3'])
    fout.write(f" Pearson r = {r:.3f},  p = {p:.4g}\n")

    low = df[df.absences <= 5]['G3']
    high = df[df.absences >= 15]['G3']
    t, p2 = stats.ttest_ind(low, high, equal_var=False)
    fout.write(f" Welch t-test (<=5 vs >=15 absences): "
               f"t = {t:.2f}, p = {p2:.4g}\n\n")

    fig, ax = plt.subplots(figsize=(4.2, 3))
    sns.scatterplot(df, x='absences', y='G3', alpha=0.6, ax=ax)
    sns.regplot(df, x='absences', y='G3',
                scatter=False, color='red', ax=ax)
    ax.set_xlim(-1)
    ax.set_xlabel("Number of absences")
    ax.set_ylabel("Final grade (G3)")
    ax.set_title("absences vs grade")
    save_figure(fig, Path("figures/fig3_absences_scatter.png"))


def question_3_family(df, fout):
    """Q3 – Family educational support vs grade."""
    fout.write("Q3: Family support vs grades\n")
    yes = df[df.famsup == 'yes']['G3']
    no  = df[df.famsup == 'no']['G3']
    t, p = stats.ttest_ind(yes, no, equal_var=False)
    diff = yes.mean() - no.mean()
    fout.write(f" Mean(G3) yes = {yes.mean():.2f}, "
               f"no = {no.mean():.2f} (diff = {diff:.2f}); "
               f"t = {t:.2f}, p = {p:.4g}\n\n")

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.boxplot(data=df, x='famsup', y='G3', palette='Set2', ax=ax)
    ax.set_xlabel("Family educational support")
    ax.set_ylabel("Final grade (G3)")
    ax.set_title("Grades by family support")
    save_figure(fig, Path("figures/fig4_family_support.png"))


def question_4_romantic(df, fout):
    """Q4 - Romantic relationship vs grade."""
    fout.write("Q4: Romantic relationship vs grades\n")
    yes = df[df.romantic == 'yes']['G3']
    no  = df[df.romantic == 'no']['G3']
    t, p = stats.ttest_ind(yes, no, equal_var=False)
    diff = yes.mean() - no.mean()
    fout.write(f" Mean(G3) in-relationship = {yes.mean():.2f}, "
               f"single = {no.mean():.2f} (diff = {diff:.2f}); "
               f"t = {t:.2f}, p = {p:.4g}\n\n")

    fig, ax = plt.subplots(figsize=(3.8, 3))
    sns.boxplot(data=df, x='romantic', y='G3',
                palette='Pastel1', ax=ax)
    ax.set_xlabel("Romantic relationship")
    ax.set_ylabel("Final grade (G3)")
    ax.set_title("Grades by romantic status")
    save_figure(fig, Path("figures/fig5_romantic.png"))


def question_5_health(df, fout):
    """Q5 - health rating vs grade."""
    fout.write("Q5: health rating vs grades\n")
    rho, p = stats.spearmanr(df['health'], df['G3'])
    fout.write(f" Spearman ρ = {rho:.3f},  p = {p:.4g}\n")

    model = smf.ols('G3 ~ C(health)', data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    fout.write(" ANOVA health:\n")
    fout.write(anova.to_string()); fout.write("\n\n")

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.violinplot(df, x='health', y='G3', palette='Greens', ax=ax)
    ax.set_xlabel("Self-reported health (1=bad … 5=excellent)")
    ax.set_ylabel("Final grade (G3)")
    ax.set_title("Grades by health level")
    save_figure(fig, Path("figures/fig6_health.png"))


def statsmodels_discrete_roc(y_true, y_score):
    """Return FPR, TPR arrays (uses sklearn for convenience)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return fpr, tpr, thr


def question_6_classification(df, fout):
    """Q6 - Logistic regression predicting failure (<10)."""
    fout.write("Q6: Logistic regression - predicting failure\n")
    df = df.copy()
    df['fail'] = (df['G3'] < 10).astype(int)

    formula = "fail ~ Walc + absences + famsup + health + romantic"
    logit = smf.logit(formula, data=df).fit(disp=False)

    fout.write(logit.summary().as_text()); fout.write("\n")
    preds = logit.predict(df)
    fpr, tpr, _ = statsmodels_discrete_roc(df['fail'], preds)
    auc = np.trapz(tpr, fpr)
    fout.write(f" ROC AUC = {auc:.3f}\n\n")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], '--', color='grey')
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC - Fail prediction")
    ax.legend()
    save_figure(fig, Path("figures/fig7_ROC_curve.png"))

    # Confusion matrix at 0.5 threshold
    y_pred = (preds >= 0.5).astype(int)
    cm = pd.crosstab(df['fail'], y_pred,
                     rownames=['Actual'], colnames=['Pred'])
    fout.write("Confusion matrix (threshold 0.5):\n")
    fout.write(cm.to_string()); fout.write("\n\n")


QUESTION_FUNCS = {
    "Q1": question_1_alcohol,
    "Q2": question_2_absences,
    "Q3": question_3_family,
    "Q4": question_4_romantic,
    "Q5": question_5_health,
    "Q6": question_6_classification,
}

# --------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------- #
def main(selected_questions: list[str], seed: int):
    np.random.seed(seed)

    Path("figures").mkdir(exist_ok=True)

    df = load_and_concat(Path("data/student-mat.csv"), Path("data/student-por.csv"))


    with open("results_summary.txt", "w", encoding='utf-8') as fout:
        # descriptive_stats(df, fout)
        for q in selected_questions:
            print(f"\n=== Running {q} ===")
            QUESTION_FUNCS[q](df, fout)

    print("\n[DONE] Analyses complete.  "
          "Figures → ./figures/   |   Summary → results_summary.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""\
            Run statistical analyses for Student Alcohol dataset.
            Available questions:
              Q1 - Alcohol vs grades
              Q2 - absences vs grades
              Q3 - Family support
              Q4 - Romantic relationship
              Q5 - health status
              Q6 - Classification (fail prediction)
            """))
    parser.add_argument("--questions", type=str, default="ALL",
                        help="Comma-sep list of question codes (e.g., Q1,Q3) "
                             "or 'ALL' (default)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    args = parser.parse_args()

    if args.questions.upper() == "ALL":
        qlist = list(QUESTION_FUNCS.keys())
    else:
        qlist = [q.strip().upper() for q in args.questions.split(',')]
        unknown = set(qlist) - set(QUESTION_FUNCS.keys())
        if unknown:
            raise ValueError(f"Unknown question code(s): {', '.join(unknown)}")

    main(qlist, args.seed)
