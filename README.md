# Streamlit Caselet - Finance Portal

# Overview
Design a Financial Analysis Portal web application using Streamlit that enables users to perform comprehensive financial assessments of companies. The portal should allow users to input one or more company tickers for analysis, and subsequently generate a comparative report containing key financial metrics. The generated report should facilitate an in-depth evaluation of the financial health of the selected companies, aiding users in making informed decisions. The application should also include interactive visualizations that effectively illustrate the financial data, assisting clients in gaining a better understanding of the comparative financial performance of the companies.

# Objective
Develop an application to input company tickers, generate comparative reports, and visualize financial metrics for informed decision-making.

# Key Features
- Input company tickers for analysis.
- Generate comparative financial reports.
- Interactive visualizations for better data understanding.

# User Workflow
###### 1. Input Company Tickers**
 - Users enter the company tickers they want to analyze.
 - Tickers act as identifiers for companies.

###### 2. Generate Comparative Report
 - Application fetches financial data for selected companies.
 - Generates a comprehensive report with key financial metrics.

###### 3. Interactive Visualizations
- Visualizations illustrate financial data.
- Aid users in understanding comparative financial performance.
- Enhance data interpretation and decision-making.

# Key Components
- User Input Interface: Company tickers input field.
- Data Retrieval Module: Fetches financial data for analysis and makes it available for download.
- Comparative Analysis Engine: Calculates financial metrics and generates reports.
- Visualization Engine: Creates interactive visualizations for better data representation.

# App Screenshots
![App Screenshot](https://github.com/bharathngowda/yahoo_finance_clone/blob/main/screenshots/1.png)
![App Screenshot](https://github.com/bharathngowda/yahoo_finance_clone/blob/main/screenshots/2.png) 
![App Screenshot](https://github.com/bharathngowda/yahoo_finance_clone/blob/main/screenshots/3.png)
![App Screenshot](https://github.com/bharathngowda/yahoo_finance_clone/blob/main/screenshots/4.png)
![App Screenshot](https://github.com/bharathngowda/yahoo_finance_clone/blob/main/screenshots/5.png)
![App Screenshot](https://github.com/bharathngowda/yahoo_finance_clone/blob/main/screenshots/6.png)
![App Screenshot](https://github.com/bharathngowda/yahoo_finance_clone/blob/main/screenshots/7.png)
![App Screenshot](https://github.com/bharathngowda/yahoo_finance_clone/blob/main/screenshots/8.png) 
![App Screenshot](https://github.com/bharathngowda/yahoo_finance_clone/blob/main/screenshots/9.png)

## Programming Language and Packages

The app is built using Python 3.10+.

The main Packages used are - 
- plotly - to make the charts
- streamlit - to build the interactive web app
- yahooquery - to make requests for getting stock data for the selected tickers
- pandas and numpy - for data processing and cleaning.


### Installation

To run this notebook interactively:

1. Download this repository in a zip file by clicking on this [link](https://github.com/bharathngowda/COVID_19_Dashboard/archive/refs/heads/master.zip) or execute this from the terminal:
`https://github.com/bharathngowda/COVID_19_Dashboard.git`

2. Install [virtualenv](http://virtualenv.readthedocs.org/en/latest/installation.html)
3. Navigate to the directory where you unzipped or cloned the repo and create a virtual environment with `python -m venv venv` from the command line or terminal
4. Activate the environment with `venv/Scripts/activate.bat` from the command line or terminal
5. Install the required dependencies with `pip install -r requirements.txt`from the command line or terminal
6. Navigate to the app folder with `cd app` from the command line or terminal
7. Execute `streamlit run app.py` from the command line or terminal
8. Copy the link `http://localhost:8888` from command prompt and paste in browser and the app will load
9. When you're done deactivate the virtual environment with `deactivate`.
