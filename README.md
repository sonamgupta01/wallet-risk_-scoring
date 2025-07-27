Wallet Risk Scoring from Scratch 🔐📊

This project aims to compute risk scores for 100 Ethereum wallets by analyzing their on-chain activity on Compound V2 and V3 protocols. The scoring system ranges from 0 to 1000, where higher scores indicate higher risk. A combination of feature engineering, on-chain data parsing, and a rule-based model was used.

## How to Run This Project 🚀

1. **Set up Python Environment**
   ```bash
   python -m venv myenv
   myenv\Scripts\activate   # For Windows
   pip install -r requirements.txt
    python risk_scoring.py
   
2. Outputs will be saved inside the output/ folder:

detailed_analysis.csv: Full feature breakdown per wallet.
risk_scores.csv: Final output containing wallet_id and corresponding score.

<img width="1017" height="461" alt="image" src="https://github.com/user-attachments/assets/987c4feb-2198-450a-84de-3b018f274de6" />
<img width="713" height="602" alt="image" src="https://github.com/user-attachments/assets/f81657c4-5203-4c80-9ee5-742ed6637e14" />

