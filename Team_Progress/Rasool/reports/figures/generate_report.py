import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, sys
sys.path.insert(0, '.')
from tests.Files.src.db.connection import get_engine
engine = get_engine()
df = pd.read_sql('SELECT * FROM customers_clean', con=engine)
# Create a multi-panel summary figure
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Telco Customer Churn — EDA Summary Dashboard',
             fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
# Panel 1: Churn distribution
ax1 = fig.add_subplot(gs[0, 0])
df['churn'].value_counts().plot.pie(ax=ax1, autopct='%1.1f%%',
    colors=['#4CAF50','#F44336'], labels=['Stayed','Churned'])
ax1.set_title('Churn Distribution', fontweight='bold')
ax1.set_ylabel('')
# Panel 2: Tenure by churn
ax2 = fig.add_subplot(gs[0, 1])
for val, color, label in [(0,'#4CAF50','Stayed'),(1,'#F44336','Churned')]:
    ax2.hist(df[df['churn']==val]['tenure'], bins=25,
             alpha=0.6, color=color, label=label)
ax2.set_title('Tenure Distribution', fontweight='bold')
ax2.set_xlabel('Tenure (months)')
ax2.legend()
# Panel 3: Monthly charges
ax3 = fig.add_subplot(gs[0, 2])
for val, color, label in [(0,'#4CAF50','Stayed'),(1,'#F44336','Churned')]:
    ax3.hist(df[df['churn']==val]['monthlycharges'], bins=30,
             alpha=0.6, color=color, label=label)
ax3.set_title('Monthly Charges', fontweight='bold')
ax3.legend()
# Panel 4: Contract churn rate
ax4 = fig.add_subplot(gs[1, 0:2])
cr = df.groupby('contract')['churn'].mean() * 100
cr.plot.bar(ax=ax4, color=['#2196F3','#FF9800','#4CAF50'], rot=0)
ax4.set_title('Churn Rate by Contract Type', fontweight='bold')
ax4.set_ylabel('Churn Rate (%)')
for p in ax4.patches:
    ax4.annotate(f'{p.get_height():.1f}%',
        (p.get_x()+p.get_width()/2, p.get_height()+0.5),
        ha='center', fontsize=10)
# Panel 5: Correlation heatmap
ax5 = fig.add_subplot(gs[1, 2])
num_cols = ['tenure','monthlycharges','totalcharges','seniorcitizen','churn']
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
            cmap='RdBu_r', ax=ax5, center=0)
ax5.set_title('Correlation Matrix', fontweight='bold')
# Panel 6: Key metrics table
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')
metrics = [
    ['Metric', 'Value', 'Insight'],
    ['Total Customers', '7,043', 'Sufficient for ML modelling'],
    ['Churn Rate', '26.5%', 'Imbalanced — use class_weight or SMOTE'],
    ['Avg Tenure (Churners)', f"{df[df['churn']==1]['tenure'].mean():.0f} months", 'Much
 shorter than retained customers'],
    ['Avg Monthly Charge (Churners)', f"${df[df['churn']==1]['monthlycharges'].mean():.2
f}", 'Higher charges correlate with churn'],
    ['Month-to-Month Churn Rate', '~43%', 'Primary retention target'],
    ['Electronic Check Churn Rate', '~45%', 'Promote auto-payment options'],
]
t = ax6.table(cellText=metrics[1:], colLabels=metrics[0],
              cellLoc='center', loc='center',
              bbox=[0, 0, 1, 1])
t.auto_set_font_size(False)
t.set_fontsize(10)
t.auto_set_column_width([0,1,2])
for (r,c), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1A237E')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#E8EAF6')
ax6.set_title('Key EDA Findings', fontweight='bold', pad=15)
plt.savefig('reports/figures/00_eda_summary_dashboard.png',
            dpi=150, bbox_inches='tight')
plt.show()
print('Saved: reports/figures/00_eda_summary_dashboard.png')