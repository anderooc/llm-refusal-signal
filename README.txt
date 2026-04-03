/* THIS CODE IS MY OWN WORK . IT WAS WRITTEN WITHOUT
 CONSULTING CODE WRITTEN BY OTHER STUDENTS OR RELYING
 SOLELY ON LARGE LANGUAGE MODELS SUCH AS CHATGPT .
Andrew Chang */

Installation Instructions
1. Clone or unzip the project directory
Ensure the following files are present:
- refusal_signal_extraction.ipynb
- requirements.txt
- README.txt

2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

3. Run all cells of jupyter notebook
3.1 Launch jupyter notebook if needed:
jupyter notebook


Expected Outputs
After running the notebook, you should observe the following:

1. Console Outputs
- Printed system prompt containing hidden rules (for verification)
- Progress logs for single turn and multiturn experiments
- Summary statistics for keyword and semantic leakage
- Reconstructed policy text printed to the console

2. Generated Figures
Bar plots showing:
- Keyword leakage rate by probing category
- Average semantic leakage score by probing category

These figures visualize how different probing strategies affect leakage.

3. Generated Files (Saved Automatically)
A new directory named results/ will be created containing:
- results/raw_responses.csv
- results/aggregate_metrics.csv
- results/reconstructed_policies.txt