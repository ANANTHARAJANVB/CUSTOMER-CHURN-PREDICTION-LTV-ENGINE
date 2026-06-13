import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
sys.path.insert(0, '.')
from tests.Files.src.db.connection import get_engine
#  Setup 
os.makedirs('reports/figures', exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)
engine = get_engine()
df = pd.read_sql('SELECT * FROM customers', con=engine)
# Convert TotalCharges to numeric (has some empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f'Loaded {len(df):,} rows')
#  Plot 1: Churn Distribution (Pie + Bar) 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
churn_counts = df['Churn'].value_counts()
# Pie chart
axes[0].pie(churn_counts, labels=churn_counts.index,
            autopct='%1.1f%%', colors=['#2196F3','#F44336'],
            startangle=90, explode=(0.05, 0.05))
axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
# Bar chart
sns.countplot(x='Churn', data=df, ax=axes[1],
              palette=['#2196F3','#F44336'])
axes[1].set_title('Churn Count', fontsize=14, fontweight='bold')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height()):,}',
        (p.get_x()+p.get_width()/2., p.get_height()),
        ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/01_churn_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: 01_churn_distribution.png')
#  Plot 2: Tenure Distribution by Churn 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Histogram
for churn_val, color in [('No','#2196F3'),('Yes','#F44336')]:
    axes[0].hist(df[df['Churn']==churn_val]['tenure'],
                 bins=30, alpha=0.6, label=f'Churn={churn_val}', color=color)
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Customer Count')
axes[0].set_title('Tenure Distribution by Churn Status')
axes[0].legend()
# Box plot
sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[1],
            palette=['#2196F3','#F44336'])
axes[1].set_title('Tenure Spread by Churn')
plt.tight_layout()
plt.savefig('reports/figures/02_tenure_by_churn.png', dpi=150, bbox_inches='tight')
plt.show()
#  Plot 3: Monthly Charges Distribution 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn',
             bins=40, ax=axes[0], palette=['#2196F3','#F44336'])
axes[0].set_title('Monthly Charges Distribution')
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[1],
            palette=['#2196F3','#F44336'])
axes[1].set_title('Monthly Charges by Churn')
plt.tight_layout()
plt.savefig('reports/figures/03_monthly_charges.png', dpi=150, bbox_inches='tight')
plt.show()
#  Plot 4: Churn Rate by Contract Type 
fig, ax = plt.subplots(figsize=(10, 5))
churn_contract = df.groupby('Contract')['Churn'].apply(
    lambda x: (x=='Yes').mean() * 100).reset_index()
churn_contract.columns = ['Contract', 'ChurnRate']
sns.barplot(x='Contract', y='ChurnRate', data=churn_contract,
            palette='Reds_d', ax=ax)
ax.set_ylabel('Churn Rate (%)')
ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
        (p.get_x()+p.get_width()/2., p.get_height()),
        ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('reports/figures/04_churn_by_contract.png', dpi=150, bbox_inches='tight')
plt.show()
#  Plot 5: Correlation Heatmap (numeric features) 
numeric_cols = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen']
df_num = df[numeric_cols].copy()
df_num['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)
corr = df_num.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            square=True, linewidths=0.5)
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('=== EDA Part 1 Complete — 5 charts saved to reports/figures/ ===')