Portfolio Optimization App
--------------------------


--------------------------------------------------------------------------------
Project Structure
--------------------------------------------------------------------------------
my_portfolio_app/
├── app.py
├── assets/
│   └── styles.css       
├── requirements.txt     # Python dependencies
├── utils/
│   ├── data_processing.py  # Data fetching/cleaning with yfinance
│   ├── optimization.py     # Markowitz optimization with PyPortfolioOpt
│   └── plotting.py         # Plot creation for frontier, random portfolios, etc.
└── README.md

- app.py: Main Dash application (layout, callbacks, etc.)
- assets/styles.css: Custom CSS that Dash automatically loads.
- requirements.txt: Lists the required Python libraries.
- utils/:
  - data_processing.py: Functions to fetch data from Yahoo Finance and extract close prices.
  - optimization.py: Functions to compute random portfolios, the efficient frontier, max Sharpe, and min volatility.
  - plotting.py: Plotly-based charts for the frontier and more.

--------------------------------------------------------------------------------
Installation & Setup
--------------------------------------------------------------------------------
1. Clone or Download the Repository

  
2. Create and Activate a Virtual Environment

   python -m portfolio_venv venv

   - Windows:
       portfolio_venv\Scripts\activate

   - macOS/Linux:
       source portfolio_venv/bin/activate

3. Install Dependencies

   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

4. Run the App

   python app.py

   You’ll see output like:
     Dash is running on http://127.0.0.1:8050/

5. Open in Browser

   Open http://127.0.0.1:8050 in your browser to access the Portfolio Optimization Tool.
