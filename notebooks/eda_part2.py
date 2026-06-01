import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, sys
sys.path.insert(0, '.')
from src.db.connection import get_engine
os.makedirs('reports/figures', exist_ok=True)
sns.set_theme(style='whitegrid')
engine = get_engine()
df = pd.read_sql('SELECT * FROM customers', con=engine)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Helper: churn rate for a column
def churn_rate(df, col):
    return df.groupby(col)['Churn'].apply(
        lambda x: round((x=='Yes').mean()*100, 1))
#  Plot 1: Churn Rate by Internet Service 
fig, axes = plt.subplots(1, 2, figsize=(14,5))
sns.countplot(x='InternetService', hue='Churn', data=df, ax=axes[0],
              palette=['#2196F3','#F44336'])
axes[0].set_title('Internet Service Count by Churn')
cr = churn_rate(df, 'InternetService').reset_index()
cr.columns = ['InternetService', 'ChurnRate']
sns.barplot(x='InternetService', y='ChurnRate', data=cr,
            ax=axes[1], palette='Oranges_d')
axes[1].set_title('Churn Rate % by Internet Service')
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.1f}%',
        (p.get_x()+p.get_width()/2., p.get_height()),
        ha='center', va='bottom')
plt.tight_layout()
plt.savefig('reports/figures/06_internet_service_churn.png', dpi=150, bbox_inches='tight
')
plt.show()
#  Plot 2: Churn Rate by Payment Method 
fig, ax = plt.subplots(figsize=(12, 5))
cr2 = churn_rate(df, 'PaymentMethod').reset_index()
cr2.columns = ['PaymentMethod', 'ChurnRate']
cr2 = cr2.sort_values('ChurnRate', ascending=False)
sns.barplot(x='PaymentMethod', y='ChurnRate', data=cr2,
            palette='Reds_d', ax=ax)
ax.set_title('Churn Rate by Payment Method', fontsize=13, fontweight='bold')
ax.set_xlabel('')
plt.xticks(rotation=15)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
        (p.get_x()+p.get_width()/2., p.get_height()),
        ha='center', va='bottom')
plt.tight_layout()
plt.savefig('reports/figures/07_payment_method_churn.png', dpi=150, bbox_inches='tight')
plt.show()
#  Plot 3: Add-on services churn comparison 
addons = ['OnlineSecurity','OnlineBackup','DeviceProtection',
          'TechSupport','StreamingTV','StreamingMovies']
addon_churn = []
for addon in addons:
    no_service = df[df[addon]=='No']['Churn'].eq('Yes').mean()*100
    yes_service = df[df[addon]=='Yes']['Churn'].eq('Yes').mean()*100
    addon_churn.append({'Service': addon, 'Without': round(no_service,1),
                        'With': round(yes_service,1)})
addon_df = pd.DataFrame(addon_churn)
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(addon_df))
w = 0.35
ax.bar(x-w/2, addon_df['Without'], w, label='Without Service',
       color='#F44336', alpha=0.85)
ax.bar(x+w/2, addon_df['With'], w, label='With Service',
       color='#4CAF50', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(addon_df['Service'], rotation=20, ha='right')
ax.set_ylabel('Churn Rate (%)')
ax.set_title('Churn Rate: With vs Without Add-on Services', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/08_addon_services_churn.png', dpi=150, bbox_inches='tight')
plt.show()
#  Plot 4: Senior Citizen churn rate 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['SeniorCitizen_Label'] = df['SeniorCitizen'].map({0:'Not Senior',1:'Senior'})
sns.countplot(x='SeniorCitizen_Label', hue='Churn', data=df,
              ax=axes[0], palette=['#2196F3','#F44336'])
axes[0].set_title('Senior Citizen Count')
cr4 = churn_rate(df, 'SeniorCitizen_Label').reset_index()
cr4.columns = ['Senior', 'ChurnRate']
sns.barplot(x='Senior', y='ChurnRate', data=cr4,
            ax=axes[1], palette='Purples_d')
axes[1].set_title('Churn Rate: Senior vs Non-Senior')
plt.tight_layout()
plt.savefig('reports/figures/09_senior_citizen_churn.png', dpi=150, bbox_inches='tight')
plt.show()
#  Summary Report 
print('\n' + '='*60)
print('EDA PART 2 — SUMMARY INSIGHTS')
print('='*60)
print(f'Fiber optic churn: {churn_rate(df,"InternetService")["Fiber optic"]:.1f}%')
print(f'DSL churn:         {churn_rate(df,"InternetService")["DSL"]:.1f}%')
print(f'No internet churn: {churn_rate(df,"InternetService")["No"]:.1f}%')
print()
print('Payment method churn rates:')
for pm, rate in churn_rate(df,'PaymentMethod').items():
    print(f'  {pm}: {rate}%')
print()
print('Key finding: Customers WITHOUT security/backup services churn much more!')
print('='*60)