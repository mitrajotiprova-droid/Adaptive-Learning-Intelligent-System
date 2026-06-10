
"""
╔══════════════════════════════════════════════════════════════════╗
║     STUDENT PERFORMANCE PREDICTION — FULL ML PROJECT            ║
║     Features: EDA · Preprocessing · Model Comparison ·          ║
║               SMOTE · Tuning · SHAP · ROC · Prediction App       ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     cross_val_score, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score,
                              accuracy_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

print("✅ All imports successful\n")




if 'results' not in globals() or not isinstance(results, dict):
    print("Error: The 'results' dictionary (containing model data) has not been defined or is not in the expected format.")
    print("Please ensure you have executed 'STEP 4: MODEL COMPARISON' (cell 3MajGuEvuBCK) before running this cell.")
else:
    for name, info in results.items():
        model = info["model"]
        print("\n" + "="*50)
        print(f"MODEL: {name}")
        print("="*50)

        if hasattr(model, "coef_"):
            print("Weights:", model.coef_)

        if hasattr(model, "intercept_"):
            print("Intercept:", model.intercept_)

        if hasattr(model, "feature_importances_"):
            print("Feature Importance:", model.feature_importances_)

        if hasattr(model, "estimators_"):
            print("Number of Trees:", len(model.estimators_))

        if hasattr(model, "support_vectors_"):
            print("Support Vectors shape:", model.support_vectors_.shape)


# ─────────────────────────────────────────────
# 1. LOAD & INSPECT DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: DATA LOADING & INSPECTION")
print("=" * 60)

df = pd.read_csv("/content/alis_final_fixed.csv")

print(f"📦 Dataset Shape: {df.shape}")
print(f"\n🔍 First 5 rows:\n{df.head()}")
print(f"\n📊 Data Types:\n{df.dtypes}")
print(f"\n❓ Missing Values:\n{df.isnull().sum()}")
print(f"\n📈 Statistical Summary:\n{df.describe()}")

# Class distribution
print(f"\n🎯 Target Distribution (pass_fail):")
print(df['pass_fail'].value_counts())
print(f"Class Balance Ratio: {df['pass_fail'].value_counts(normalize=True).round(3).to_dict()}")

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

sns.set_theme(style="whitegrid", palette="muted")

# 2a. Feature distributions split by class
numeric_features = ['attendance_percentage', 'study_hours_per_day',
                    'burnout_score', 'programming_score',
                    'sleep_hours', 'daily_completion_percent']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, feat in zip(axes.flatten(), numeric_features):
    for label, grp in df.groupby('pass_fail'):
        grp[feat].hist(bins=25, ax=ax, alpha=0.6,
                       label=f"{'Pass' if label == 1 else 'Fail'}")
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.legend()
plt.suptitle('Feature Distributions by Pass/Fail', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150)
plt.show()
print("✅ EDA plot saved: eda_distributions.png")
# 2b. Correlation heatmap
numeric_df = df.select_dtypes(include='number').drop(
    columns=[c for c in ['student_id'] if c in df.columns])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print("✅ Heatmap saved: correlation_heatmap.png")


# 2c. Boxplots — outlier detection
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, feat in zip(axes.flatten(), numeric_features):
    df.boxplot(column=feat, by='pass_fail', ax=ax)
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel('0 = Fail, 1 = Pass')
plt.suptitle('Boxplots by Class (Outlier Detection)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('boxplots.png', dpi=150)
plt.show()
print("✅ Boxplots saved: boxplots.png")
# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: DATA PREPROCESSING")
print("=" * 60)

df_proc = df.copy()

# Drop ID column if present
if 'student_id' in df_proc.columns:
    df_proc.drop(columns=['student_id'], inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df_proc.select_dtypes(include='object').columns:
    df_proc[col] = le.fit_transform(df_proc[col])
    print(f"  🔤 Encoded: {col}")

# Outlier removal via IQR on numeric features
num_cols = [c for c in df_proc.select_dtypes(include=['int64', 'float64']).columns
            if c != 'pass_fail']

df_clean = df_proc.copy()
for col in num_cols:
    Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[(df_clean[col] >= Q1 - 1.5*IQR) &
                        (df_clean[col] <= Q3 + 1.5*IQR)]

print(f"\n  Original rows : {df_proc.shape[0]}")
print(f"  After cleaning: {df_clean.shape[0]}")
print(f"  Rows removed  : {df_proc.shape[0] - df_clean.shape[0]}")

# Features & target
X = df_clean.drop(columns=['pass_fail'])
y = df_clean['pass_fail']

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── SMOTE: handle class imbalance ──
print(f"\n  Class counts before SMOTE: {dict(y_train.value_counts())}")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
print(f"  Class counts after  SMOTE: {dict(pd.Series(y_train_sm).value_counts())}")
print("✅ Preprocessing complete")



# ─────────────────────────────────────────────
# 4. MODEL COMPARISON (5 algorithms)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: MODEL COMPARISON")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost"            : XGBClassifier(use_label_encoder=False,
                                         eval_metric='logloss', random_state=42),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM"                : SVC(probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, mdl in models.items():
    cv_scores = cross_val_score(mdl, X_train_sm, y_train_sm,
                                cv=cv, scoring='f1')
    mdl.fit(X_train_sm, y_train_sm)
    y_pred  = mdl.predict(X_test_sc)
    y_prob  = mdl.predict_proba(X_test_sc)[:, 1]

    results[name] = {
        "CV F1 (mean)": cv_scores.mean(),
        "CV F1 (std)" : cv_scores.std(),
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test F1"      : f1_score(y_test, y_pred),
        "ROC-AUC"      : roc_auc_score(y_test, y_prob),
        "Precision"    : precision_score(y_test, y_pred),
        "Recall"       : recall_score(y_test, y_pred),
        "model"        : mdl,
        "y_prob"       : y_prob,
        "y_pred"       : y_pred,
    }
    print(f"  ✔ {name:<25} CV-F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} "
          f"| Test-AUC: {roc_auc_score(y_test, y_prob):.3f}")

# Summary table
results_df = pd.DataFrame({k: {m: v for m, v in v.items()
                                if m not in ('model','y_prob','y_pred')}
                            for k, v in results.items()}).T
print(f"\n📊 Model Comparison Table:\n{results_df.round(4).to_string()}")

# Best model
best_name = results_df['ROC-AUC'].astype(float).idxmax()
best       = results[best_name]
print(f"\n🏆 Best Model: {best_name}  (AUC = {best['ROC-AUC']:.4f})")


# ─────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING (best model)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 5: HYPERPARAMETER TUNING — {best_name}")
print("=" * 60)

if best_name == "Random Forest":
    param_grid = {'n_estimators': [100, 200, 300],
                  'max_depth'   : [None, 10, 20],
                  'min_samples_split': [2, 5]}
    tuned_base = RandomForestClassifier(random_state=42)

elif best_name == "XGBoost":
    param_grid = {'n_estimators': [100, 200],
                  'max_depth'   : [3, 5, 7],
                  'learning_rate': [0.05, 0.1, 0.2]}
    tuned_base = XGBClassifier(use_label_encoder=False,
                               eval_metric='logloss', random_state=42)

elif best_name == "Gradient Boosting":
    param_grid = {'n_estimators': [100, 200],
                  'max_depth'   : [3, 5],
                  'learning_rate': [0.05, 0.1]}
    tuned_base = GradientBoostingClassifier(random_state=42)

else:  # fallback
    param_grid = {'C': [0.1, 1, 10]}
    tuned_base = LogisticRegression(max_iter=1000, random_state=42)

grid_search = GridSearchCV(tuned_base, param_grid,
                           cv=5, scoring='roc_auc',
                           n_jobs=-1, verbose=0)
grid_search.fit(X_train_sm, y_train_sm)

best_model  = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_sc)
y_prob_best = best_model.predict_proba(X_test_sc)[:, 1]

print(f"  Best Params : {grid_search.best_params_}")
print(f"  Tuned AUC   : {roc_auc_score(y_test, y_prob_best):.4f}")
print(f"  Tuned F1    : {f1_score(y_test, y_pred_best):.4f}")
print(f"\n📋 Detailed Classification Report:\n"
      f"{classification_report(y_test, y_pred_best, target_names=['Fail','Pass'])}")

# ─────────────────────────────────────────────
# 6. VISUALISATIONS — ROC, Confusion Matrix, Feature Importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: RESULTS VISUALISATION")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── 6a. ROC curves (all models) ──
ax1 = fig.add_subplot(gs[0, :2])
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax1.plot(fpr, tpr, lw=2,
             label=f"{name}  (AUC={res['ROC-AUC']:.3f})")
ax1.plot([0,1],[0,1],'k--', lw=1)
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves — All Models', fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)



# ── 6b. Bar chart — model comparison ──
ax2 = fig.add_subplot(gs[0, 2])
metric_names = ["Test Accuracy", "Test F1", "ROC-AUC"]
x = np.arange(len(metric_names))
width = 0.15
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metric_names]
    ax2.bar(x + i*width, vals, width, label=name)
ax2.set_xticks(x + width*2)
ax2.set_xticklabels(metric_names, fontsize=9)
ax2.set_ylim(0.5, 1.05)
ax2.set_title('Model Comparison', fontweight='bold')
ax2.legend(fontsize=7)



# ── 6c. Confusion matrix (best tuned model) ──
ax3 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail','Pass'],
            yticklabels=['Fail','Pass'], ax=ax3)
ax3.set_title(f'Confusion Matrix\n({best_name} — Tuned)', fontweight='bold')
ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')

# ── 6d. Feature importances ──
ax4 = fig.add_subplot(gs[1, 1:])
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_,
                            index=X.columns).sort_values(ascending=True)
    importances.plot(kind='barh', ax=ax4, color='steelblue')
    ax4.set_title('Feature Importances (Tuned Model)', fontweight='bold')
    ax4.set_xlabel('Importance Score')
else:
    ax4.text(0.5, 0.5, 'Feature importance\nnot available for this model',
             ha='center', va='center', transform=ax4.transAxes)

plt.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold', y=1.01)
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Model evaluation dashboard saved: model_evaluation.png")

# ─────────────────────────────────────────────
# 7. SHAP — MODEL EXPLAINABILITY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: SHAP EXPLAINABILITY")
print("=" * 60)

try:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_sc)

    # For binary classifiers shap_values may be a list
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test_sc,
                      feature_names=X.columns.tolist(),
                      show=False)
    plt.title('SHAP Summary Plot — Feature Impact', fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150)
    plt.show()
    print("✅ SHAP plot saved: shap_summary.png")

    # SHAP bar (global importance)
    plt.figure(figsize=(10, 5))
    shap.summary_plot(sv, X_test_sc,
                      feature_names=X.columns.tolist(),
                      plot_type='bar', show=False)
    plt.title('SHAP Feature Importance (Global)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_importance.png', dpi=150)
    plt.show()
    print("✅ SHAP importance saved: shap_importance.png")

except Exception as e:
    print(f"  ⚠ SHAP skipped: {e}")

# ─────────────────────────────────────────────
# 8. LIVE PREDICTION APP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: LIVE STUDENT PREDICTION APP")
print("=" * 60)

from tabulate import tabulate

print("""
╔══════════════════════════════════════════════════╗
║         🎓 STUDENT PASS/FAIL PREDICTOR           ║
║    Powered by ML — Enter student details below   ║
╚══════════════════════════════════════════════════╝
""")

def get_float(prompt, lo=0, hi=100):
    while True:
        try:
            val = float(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  ⚠ Enter a value between {lo} and {hi}")
        except ValueError:
            print("  ⚠ Please enter a number")

attendance  = get_float("📌 Attendance Percentage (0–100): ", 0, 100)
study_hours = get_float("📚 Study Hours Per Day (0–24): ", 0, 24)
sleep_hours = get_float("😴 Sleep Hours Per Day (0–12): ", 0, 12)
screen_time = get_float("📱 Screen Time Hours (0–16): ", 0, 16)
burnout     = get_float("🔥 Burnout Score (0–10): ", 0, 10)
completion  = get_float("✅ Daily Completion % (0–100): ", 0, 100)
prog_score  = get_float("💻 Programming Score (0–100): ", 0, 100)
math_score  = get_float("📐 Math Score (0–100): ", 0, 100)
science_sc  = get_float("🔬 Science Score (0–100): ", 0, 100)
arts_sc     = get_float("🎨 Arts Score (0–100): ", 0, 100)

# Build input vector matching training features
# (adjust column order to match X.columns exactly)
input_dict = {col: 0 for col in X.columns}
mapping = {
    'attendance_percentage'  : attendance,
    'study_hours_per_day'    : study_hours,
    'sleep_hours'            : sleep_hours,
    'screen_time'            : screen_time,
    'burnout_score'          : burnout,
    'daily_completion_percent': completion,
    'programming_score'      : prog_score,
    'math_score'             : math_score,
    'science_score'          : science_sc,
    'arts_score'             : arts_sc,
}
for k, v in mapping.items():
    if k in input_dict:
        input_dict[k] = v

input_df   = pd.DataFrame([input_dict])
input_sc   = scaler.transform(input_df)

prediction = best_model.predict(input_sc)[0]
probability = best_model.predict_proba(input_sc)[0]

print("\n" + "─"*50)
print("📊 PREDICTION RESULT")
print("─"*50)
print(tabulate([
    ["Outcome"      , "✅ PASS" if prediction == 1 else "❌ FAIL"],
    ["Pass Probability" , f"{probability[1]*100:.1f}%"],
    ["Fail Probability" , f"{probability[0]*100:.1f}%"],
    ["Model Used"   , best_name],
    ["Model AUC"    , f"{roc_auc_score(y_test, y_prob_best):.4f}"],
], tablefmt='fancy_grid'))

# ── Personalised recommendations ──
print("\n⭐ PERSONALISED RECOMMENDATIONS")
print("─"*50)
recs = []
if attendance   < 75 : recs.append("📌 Boost attendance above 75% — it strongly predicts passing")
if study_hours  < 4  : recs.append("📚 Study at least 4 hrs/day — currently below threshold")
if sleep_hours  < 7  : recs.append("😴 Get 7–8 hrs sleep — lack of sleep hurts performance")
if burnout      > 6  : recs.append("🧘 Burnout is high — schedule breaks and rest days")
if completion   < 80 : recs.append("✅ Improve daily task completion to 80%+")
if prog_score   < 60 : recs.append("💻 Focus extra time on Programming — score below 60")
if math_score   < 60 : recs.append("📐 Math needs attention — score below 60")

if recs:
    for r in recs:
        print(f"  → {r}")
else:
    print("  🌟 Great profile! Keep up the excellent habits.")

# ── Study schedule ──
subjects = {"Programming": prog_score, "Math": math_score,
            "Science": science_sc,    "Arts": arts_sc}
priority = sorted(subjects, key=subjects.get)  # weakest first

deep = round(study_hours * 0.5, 1)
prac = round(study_hours * 0.3, 1)
rev  = round(study_hours * 0.2, 1)

days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
schedule = [{"Day": d,
             "Deep Study"  : f"{priority[0]} ({deep}h)",
             "Practice"    : f"{priority[1]} ({prac}h)",
             "Revision"    : f"{priority[2]} ({rev}h)",
             "Evening"     : "Assignments + Coding practice"}
            for d in days]

print("\n📅 RECOMMENDED WEEKLY STUDY SCHEDULE")
print(tabulate(pd.DataFrame(schedule), headers='keys',
               tablefmt='fancy_grid', showindex=False))

print("\n🚀 Thank you for using the Student Performance Predictor!")
print("   Developed with: Scikit-learn · XGBoost · SMOTE · SHAP\n")




