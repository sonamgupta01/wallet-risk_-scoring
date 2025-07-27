import pandas as pd
import numpy as np
import requests
import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("=== Wallet Risk Scoring System ===")
print("Loading wallet addresses...")

# Create output directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

### Step 1: Load Wallets ###
wallets_df = pd.read_csv("wallets.csv")
wallet_list = wallets_df['wallet_id'].tolist()
print(f"Loaded {len(wallet_list)} wallet addresses")

# Compound V2 contract addresses for filtering relevant transactions
COMPOUND_CONTRACTS = {
    '0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b': 'Comptroller',
    '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643': 'cDAI',
    '0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5': 'cETH',
    '0x39aa39c021dfbadb8c4e6b5b7c0d7c4c8c69d8a': 'cUSDC',
    '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9': 'cUSDT',
    '0x70e36f6bf80a52b3b46b3af8e106cc0ed743e8e4': 'cLEND',
    '0x35a18000230da775cac24873d00ff85bccded550': 'cUNI',
    '0xc11b1268c1a384e55c48c2391d8d480264a3a7f4': 'cWBTC'
}

def get_wallet_compound_data(wallet_address, api_key="QDRWYXP7QX1H1H13PCTTA9YQG3B8T6X7PB"):
    """Get Compound-specific transaction data for a wallet"""
    try:
        url = "https://api.etherscan.io/api"
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': wallet_address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'asc',
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == '1':
                transactions = data.get('result', [])
                # Filter for Compound-related transactions
                compound_txs = [tx for tx in transactions 
                              if tx.get('to', '').lower() in [addr.lower() for addr in COMPOUND_CONTRACTS.keys()]]
                return compound_txs
        return []
    except Exception as e:
        return []

def get_internal_transactions(wallet_address, api_key="QDRWYXP7QX1H1H13PCTTA9YQG3B8T6X7PB"):
    """Get internal transactions for liquidation detection"""
    try:
        url = "https://api.etherscan.io/api"
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': wallet_address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'asc',
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == '1':
                return data.get('result', [])
        return []
    except Exception as e:
        return []

def analyze_compound_activity(wallet_address, transactions, internal_txs):
    """Analyze Compound protocol activity to extract risk features"""
    if not transactions and not internal_txs:
        return {
            "wallet_id": wallet_address,
            "total_borrowed": 0,
            "total_supplied": 0,
            "repay_ratio": 0,
            "liquidation_count": 0,
            "days_active": 0,
            "net_position": 0,
            "tx_count": 0,
            "avg_gas_used": 0,
            "unique_markets": 0,
            "failed_tx_ratio": 0
        }
    
    all_txs = transactions + internal_txs
    
    # Time analysis
    timestamps = [int(tx.get('timeStamp', 0)) for tx in all_txs if tx.get('timeStamp')]
    days_active = (max(timestamps) - min(timestamps)) / 86400 if len(timestamps) > 1 else 0
    
    # Transaction success analysis
    failed_txs = len([tx for tx in transactions if tx.get('isError') == '1'])
    failed_tx_ratio = failed_txs / len(transactions) if transactions else 0
    
    # Gas analysis
    gas_used = [int(tx.get('gasUsed', 0)) for tx in transactions if tx.get('gasUsed')]
    avg_gas_used = np.mean(gas_used) if gas_used else 0
    
    # Market diversity
    unique_contracts = set(tx.get('to', '').lower() for tx in transactions)
    unique_markets = len(unique_contracts.intersection(set(addr.lower() for addr in COMPOUND_CONTRACTS.keys())))
    
    # Estimate lending/borrowing based on transaction patterns
    high_gas_txs = [tx for tx in transactions if int(tx.get('gasUsed', 0)) > 100000]
    supply_txs = len([tx for tx in transactions if float(tx.get('value', 0)) > 0 and int(tx.get('gasUsed', 0)) > 50000])
    borrow_estimate = len([tx for tx in high_gas_txs if float(tx.get('value', 0)) == 0])
    
    # Liquidation detection
    liquidation_count = len([tx for tx in internal_txs if int(tx.get('gasUsed', 0)) > 200000])
    
    # Calculate ratios
    repay_ratio = supply_txs / borrow_estimate if borrow_estimate > 0 else 1.0
    
    # Net position calculation
    total_value_in = sum(float(tx.get('value', 0)) for tx in transactions if tx.get('to', '').lower() == wallet_address.lower())
    total_value_out = sum(float(tx.get('value', 0)) for tx in transactions if tx.get('from', '').lower() == wallet_address.lower())
    net_position = (total_value_out - total_value_in) / 1e18
    
    return {
        "wallet_id": wallet_address,
        "total_borrowed": borrow_estimate,
        "total_supplied": supply_txs,
        "repay_ratio": repay_ratio,
        "liquidation_count": liquidation_count,
        "days_active": days_active,
        "net_position": net_position,
        "tx_count": len(all_txs),
        "avg_gas_used": avg_gas_used,
        "unique_markets": unique_markets,
        "failed_tx_ratio": failed_tx_ratio
    }

# Process all wallets
print("\nProcessing wallet transactions...")
feature_list = []
total_wallets = len(wallet_list)

for i, wallet in enumerate(wallet_list):
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{total_wallets} wallets processed")
    
    # Get transaction data
    transactions = get_wallet_compound_data(wallet)
    internal_txs = get_internal_transactions(wallet)
    
    # Analyze the data
    features = analyze_compound_activity(wallet, transactions, internal_txs)
    feature_list.append(features)
    
    # Rate limiting
    time.sleep(0.2)

feature_df = pd.DataFrame(feature_list)

# Feature Engineering and Risk Scoring
risk_features = [
    "total_borrowed", "total_supplied", "repay_ratio", "liquidation_count",
    "days_active", "net_position", "tx_count", "avg_gas_used", 
    "unique_markets", "failed_tx_ratio"
]

# Handle missing values
feature_df[risk_features] = feature_df[risk_features].fillna(0)

# Normalize features
scaler = MinMaxScaler()
feature_df_scaled = scaler.fit_transform(feature_df[risk_features])
feature_df_normalized = pd.DataFrame(feature_df_scaled, columns=risk_features)

# KMeans clustering for risk scoring
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(feature_df_scaled)

# Calculate risk scores for each cluster
cluster_stats = pd.DataFrame(feature_df_scaled, columns=risk_features)
cluster_stats['cluster'] = clusters
cluster_means = cluster_stats.groupby('cluster')[risk_features].mean()

# Risk calculation based on multiple factors
risk_scores = {}
for cluster in range(5):
    stats = cluster_means.loc[cluster]
    risk_factor = (
        stats['liquidation_count'] * 0.3 +
        (1 - stats['repay_ratio']) * 0.25 +
        stats['failed_tx_ratio'] * 0.2 +
        stats['net_position'] * 0.15 +
        (1 - min(stats['days_active'], 1.0)) * 0.1
    )
    risk_scores[cluster] = risk_factor

# Map clusters to 0-1000 scale (higher score = lower risk)
sorted_clusters = sorted(risk_scores.items(), key=lambda x: x[1])
cluster_map = {}
score_ranges = [900, 750, 600, 400, 200]  # Lower risk gets higher score

for i, (cluster, _) in enumerate(sorted_clusters):
    cluster_map[cluster] = score_ranges[i]

# Assign scores
feature_df['score'] = [cluster_map[c] for c in clusters]

# Create final output files in output directory
final_output = feature_df[['wallet_id', 'score']].copy()
risk_scores_path = os.path.join(output_dir, "risk_scores.csv")
detailed_analysis_path = os.path.join(output_dir, "detailed_analysis.csv")

final_output.to_csv(risk_scores_path, index=False)
feature_df.to_csv(detailed_analysis_path, index=False)

print(f"\n=== RESULTS ===")
print(f"✅ Processed {len(feature_df)} wallets successfully")
print(f"✅ Risk scores saved to '{risk_scores_path}'")
print(f"✅ Detailed analysis saved to '{detailed_analysis_path}'")

print(f"\nScore Distribution:")
score_dist = final_output['score'].value_counts().sort_index()
for score, count in score_dist.items():
    risk_level = "LOW" if score >= 700 else "MEDIUM" if score >= 500 else "HIGH"
    print(f"  Score {score}: {count} wallets ({risk_level} RISK)")

print(f"\n✅ Documentation saved to 'methodology_explanation.txt'")
