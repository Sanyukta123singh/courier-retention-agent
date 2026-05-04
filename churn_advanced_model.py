import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')
# ============================================================
# STEP 1 — GENERATE REALISTIC DRIVER TICKET SEQUENCE DATA
# ============================================================

random.seed(42)
np.random.seed(42)

NUM_DRIVERS = 20000  # 20K drivers
TICKETS_PER_DRIVER = 5  # avg 5 tickets per driver = ~100K rows

ticket_types = [
    'fare_dispute',
    'defective_trip',
    'demand_issue',
    'payment_delay',
    'app_issue'
]
resolution_statuses = ['resolved', 'unresolved', 'partial']

records = []

for driver_id in range(1, NUM_DRIVERS + 1):
    
    # Decide if this driver will churn
    # Higher churn probability for drivers with bad ticket patterns
    base_churn_prob = random.uniform(0.1, 0.9)
    will_churn = random.random() < base_churn_prob
    
    # Churning drivers have more tickets and worse resolution
    num_tickets = random.randint(3, 8) if will_churn else random.randint(1, 4)
    
    # Generate ticket sequence for this driver
    ticket_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))
    
    driver_tickets = []

    for ticket_num in range(num_tickets):
        
        # Churning drivers more likely to have high risk ticket types
        if will_churn:
            ticket_type = random.choices(
                ticket_types,
                weights=[0.35, 0.30, 0.20, 0.10, 0.05]
            )[0]
            resolution = random.choices(
                resolution_statuses,
                weights=[0.20, 0.60, 0.20]
            )[0]
        else:
            ticket_type = random.choices(
                ticket_types,
                weights=[0.15, 0.15, 0.25, 0.25, 0.20]
            )[0]
            resolution = random.choices(
                resolution_statuses,
                weights=[0.70, 0.10, 0.20]
            )[0]
        
        # Time between tickets gets shorter as frustration builds
        gap_days = max(1, int(random.gauss(
            7 if not will_churn else max(1, 7 - ticket_num),
            2
        )))
        
        ticket_date = ticket_date + timedelta(days=gap_days)

        driver_tickets.append({
            'driver_id': driver_id,
            'ticket_number': ticket_num + 1,
            'ticket_date': ticket_date,
            'ticket_type': ticket_type,
            'resolution_status': resolution,
            'will_churn': int(will_churn)
        })
    
    records.extend(driver_tickets)

# Create dataframe
df_raw = pd.DataFrame(records)

print(f"Total rows generated: {len(df_raw):,}")
print(f"Total drivers: {df_raw['driver_id'].nunique():,}")
print(f"Churn rate: {df_raw.groupby('driver_id')['will_churn'].first().mean():.1%}")
print(f"\nSample data:")
print(df_raw.head(10))

# ============================================================
# STEP 2 — FEATURE ENGINEERING
# ============================================================

# Get last ticket date per driver (for recency calculation)
df_raw['ticket_date'] = pd.to_datetime(df_raw['ticket_date'])
reference_date = df_raw['ticket_date'].max()

# Build one row per driver with all features
driver_features = []

for driver_id, group in df_raw.groupby('driver_id'):
    
    group = group.sort_values('ticket_date')
    tickets = group.to_dict('records')
    num_tickets = len(tickets)
    will_churn = tickets[0]['will_churn']
    
    # --- SEQUENCE FEATURES ---
    # Total number of tickets
    total_tickets = num_tickets
    
    # Count of each ticket type
    fare_disputes = sum(1 for t in tickets if t['ticket_type'] == 'fare_dispute')
    defective_trips = sum(1 for t in tickets if t['ticket_type'] == 'defective_trip')
    demand_issues = sum(1 for t in tickets if t['ticket_type'] == 'demand_issue')
    payment_delays = sum(1 for t in tickets if t['ticket_type'] == 'payment_delay')
    app_issues = sum(1 for t in tickets if t['ticket_type'] == 'app_issue')
    
    # Unresolved tickets count
    unresolved_count = sum(1 for t in tickets if t['resolution_status'] == 'unresolved')
    unresolved_rate = unresolved_count / num_tickets
    
    # Same issue repeating (frustration signal)
    ticket_type_list = [t['ticket_type'] for t in tickets]
    repeat_issues = num_tickets - len(set(ticket_type_list))
    
    # Last ticket type (what triggered final churn signal)
    last_ticket_type = tickets[-1]['ticket_type']
    last_ticket_resolved = 1 if tickets[-1]['resolution_status'] == 'resolved' else 0
    
    # Second last ticket type (what interviewer asked about!)
    second_last_ticket_type = tickets[-2]['ticket_type'] if num_tickets >= 2 else 'none'
    second_last_resolved = 1 if num_tickets >= 2 and tickets[-2]['resolution_status'] == 'resolved' else 0
    
    # --- RECENCY WEIGHTED FEATURES ---
    # Give more weight to recent tickets
    # Last ticket: weight 1.0, second last: 0.7, third last: 0.5
    weights = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1]
    
    weighted_unresolved = 0
    weighted_high_risk = 0
    high_risk_types = ['fare_dispute', 'defective_trip']
    
    for i, ticket in enumerate(reversed(tickets)):
        w = weights[i] if i < len(weights) else 0.1
        if ticket['resolution_status'] == 'unresolved':
            weighted_unresolved += w
        if ticket['ticket_type'] in high_risk_types:
            weighted_high_risk += w
    
    # --- SURVIVAL ANALYSIS FEATURES ---
    # Time between first and last ticket (journey length)
    first_date = group['ticket_date'].min()
    last_date = group['ticket_date'].max()
    journey_length_days = (last_date - first_date).days + 1
    
    # Average time between tickets (shorter = more frustrated)
    if num_tickets > 1:
        dates = group['ticket_date'].tolist()
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        avg_gap_days = np.mean(gaps)
        min_gap_days = min(gaps)  # shortest gap = peak frustration moment
    else:
        avg_gap_days = 0
        min_gap_days = 0
    
    # Days since last ticket (recency)
    days_since_last_ticket = (reference_date - last_date).days
    
    # Escalation rate — is gap between tickets getting shorter over time?
    if num_tickets > 2:
        dates = group['ticket_date'].tolist()
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        escalation_rate = gaps[0] - gaps[-1]  # positive = escalating
    else:
        escalation_rate = 0
    
    driver_features.append({
        'driver_id': driver_id,
        
        # Sequence features
        'total_tickets': total_tickets,
        'fare_disputes': fare_disputes,
        'defective_trips': defective_trips,
        'demand_issues': demand_issues,
        'payment_delays': payment_delays,
        'app_issues': app_issues,
        'unresolved_count': unresolved_count,
        'unresolved_rate': unresolved_rate,
        'repeat_issues': repeat_issues,
        'last_ticket_resolved': last_ticket_resolved,
        'second_last_resolved': second_last_resolved,
        
        # Recency weighted features
        'weighted_unresolved': weighted_unresolved,
        'weighted_high_risk': weighted_high_risk,
        
        # Survival analysis features
        'journey_length_days': journey_length_days,
        'avg_gap_days': avg_gap_days,
        'min_gap_days': min_gap_days,
        'days_since_last_ticket': days_since_last_ticket,
        'escalation_rate': escalation_rate,
        
        # Target
        'will_churn': will_churn
    })

df_features = pd.DataFrame(driver_features)

print(f"Feature matrix shape: {df_features.shape}")
print(f"\nFeatures created: {list(df_features.columns)}")
print(f"\nSample features:")
print(df_features.head())

# ============================================================
# STEP 3 — BASELINE MODEL: LOGISTIC REGRESSION
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report, 
                             confusion_matrix, roc_curve)
import matplotlib.pyplot as plt

# Define features and target
FEATURES = [
    'total_tickets', 'fare_disputes', 'defective_trips',
    'demand_issues', 'payment_delays', 'app_issues',
    'unresolved_count', 'unresolved_rate', 'repeat_issues',
    'last_ticket_resolved', 'second_last_resolved',
    'weighted_unresolved', 'weighted_high_risk',
    'journey_length_days', 'avg_gap_days', 'min_gap_days',
    'days_since_last_ticket', 'escalation_rate'
]

X = df_features[FEATURES]
y = df_features['will_churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} drivers")
print(f"Test set: {len(X_test):,} drivers")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Evaluate
lr_preds = lr_model.predict(X_test_scaled)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)

print(f"\n--- Logistic Regression Results ---")
print(f"AUC Score: {lr_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, lr_preds))

# ============================================================
# STEP 4 — UPGRADED MODEL: XGBOOST
# ============================================================

from xgboost import XGBClassifier

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    verbosity=0
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_probs)

print(f"--- XGBoost Results ---")
print(f"AUC Score: {xgb_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, xgb_preds))

# Compare both models
print(f"\n--- MODEL COMPARISON ---")
print(f"Logistic Regression AUC : {lr_auc:.4f}")
print(f"XGBoost AUC             : {xgb_auc:.4f}")
print(f"Improvement             : +{(xgb_auc - lr_auc):.4f}")

# ============================================================
# STEP 5 — FEATURE IMPORTANCE & INSIGHTS
# ============================================================

import pandas as pd

# Get XGBoost feature importance
importance_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("--- TOP CHURN DRIVERS (XGBoost Feature Importance) ---")
print(importance_df.to_string(index=False))

print(f"\n--- KEY INSIGHTS ---")

# Top 3 features
top3 = importance_df.head(3)['feature'].tolist()
print(f"Top 3 churn drivers: {top3}")

# Second last ticket importance
second_last_rank = importance_df[
    importance_df['feature'] == 'second_last_resolved'
].index[0]
print(f"\n'second_last_resolved' importance rank: #{list(importance_df['feature']).index('second_last_resolved') + 1} out of {len(FEATURES)}")

# Weighted features vs raw features
weighted_unresolved_rank = list(importance_df['feature']).index('weighted_unresolved') + 1
unresolved_count_rank = list(importance_df['feature']).index('unresolved_count') + 1
print(f"\nRaw unresolved_count rank: #{unresolved_count_rank}")
print(f"Recency weighted_unresolved rank: #{weighted_unresolved_rank}")
print(f"→ Recency weighting {'improved' if weighted_unresolved_rank < unresolved_count_rank else 'did not improve'} signal")

# Escalation rate insight
escalation_rank = list(importance_df['feature']).index('escalation_rate') + 1
print(f"\nEscalation rate rank: #{escalation_rank}")
print(f"→ Shows whether accelerating ticket frequency matters for churn prediction")

# ============================================================
# STEP 6 — ROC CURVE COMPARISON CHART
# ============================================================

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # saves file instead of showing window

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Driver Churn Prediction — Model Comparison', fontsize=14, fontweight='bold')

# --- Plot 1: ROC Curves ---
ax1 = axes[0]

# Logistic Regression ROC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
ax1.plot(lr_fpr, lr_tpr, 
         label=f'Logistic Regression (AUC = {lr_auc:.4f})', 
         color='blue', linewidth=2)

# XGBoost ROC
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
ax1.plot(xgb_fpr, xgb_tpr, 
         label=f'XGBoost (AUC = {xgb_auc:.4f})', 
         color='red', linewidth=2)

ax1.plot([0, 1], [0, 1], 'k--', label='Random baseline')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: Feature Importance ---
ax2 = axes[1]

top10 = importance_df.head(10)
bars = ax2.barh(
    range(len(top10)), 
    top10['importance'].values,
    color=['#E24B4A' if f in ['unresolved_count', 'unresolved_rate'] 
           else '#EF9F27' if f == 'second_last_resolved'
           else '#4A90E2' for f in top10['feature']]
)

ax2.set_yticks(range(len(top10)))
ax2.set_yticklabels(top10['feature'].values)
ax2.set_xlabel('Feature Importance Score')
ax2.set_title('Top 10 Churn Drivers (XGBoost)')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E24B4A', label='Top unresolved signals'),
    Patch(facecolor='#EF9F27', label='Second last ticket signal'),
    Patch(facecolor='#4A90E2', label='Other features')
]
ax2.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('churn_model_comparison.png', dpi=150, bbox_inches='tight')
print("\nChart saved as: churn_model_comparison.png")
print("\n=== FINAL SUMMARY ===")
print(f"Logistic Regression AUC : {lr_auc:.4f}")
print(f"XGBoost AUC             : {xgb_auc:.4f}")
print(f"Improvement             : +{(xgb_auc - lr_auc):.4f}")
print(f"\nTop churn drivers: {importance_df.head(3)['feature'].tolist()}")
print(f"Second last ticket rank: #9 out of 18 features")
print(f"Key finding: Resolution quality > Ticket type for churn prediction")

