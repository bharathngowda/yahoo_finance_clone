# import required packages
from cProfile import label
import pandas as pd
import numpy as np
import datetime
import json
import streamlit as st
from yahooquery import Ticker
import re
import plotly.express as px
import plotly.graph_objects as go
import random
import string
import warnings
warnings.filterwarnings('ignore')

# global zip code data
zip_code=pd.read_csv('input/global_zip_code_data.csv',delimiter=';')

# function
def sentence_case(string):
    if string != '':
        result = re.sub('([A-Z])', r' \1', string)
        r=result[:1].upper() + result[1:].lower()
        return r.title()
    return
def days(selected_period,max_date):
    if selected_period=='Last 30 Days':
        no_days=30
    elif selected_period=='Last 6 Months':
        no_days=180
    elif selected_period=='Year to Date':
        no_days=(max_date-datetime.date(int(datetime.datetime.today().year),1,1)).days
    elif selected_period=='Last 1 Year':
        no_days=365
    elif selected_period=='Last 5 Years':
        no_days=365*5
    return no_days

@st.cache_resource
def ticker_dataframe(ticker):
    return Ticker(ticker)

@st.cache_resource
def balance_sheet_annual_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.balance_sheet(frequency='a', trailing=True).reset_index()
    return df

@st.cache_resource
def balance_sheet_quarter_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.balance_sheet(frequency='q', trailing=True).reset_index()
    return df

@st.cache_resource
def income_statement_annual_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.income_statement(frequency='a', trailing=True).reset_index()
    return df

@st.cache_resource
def income_statement_quarter_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.income_statement(frequency='q', trailing=True).reset_index()
    return df

@st.cache_resource
def cash_flow_annual_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.cash_flow(frequency='a', trailing=True).reset_index()
    return df

@st.cache_resource
def cash_flow_quarter_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.cash_flow(frequency='q', trailing=True).reset_index()
    return df

@st.cache_resource
def stock_price_history_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.history(period='max',interval='1d').reset_index()
    df['date']=pd.to_datetime(df.date).dt.date
    df.sort_values(by=['symbol','date'],ascending=True,inplace=True,ignore_index=True)
    return df

@st.cache_resource
def profile_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.asset_profile
    return df

@st.cache_resource
def profile_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.asset_profile
    return df

@st.cache_resource
def earning_trend_func(ticker,symbol):
    ticker_df=ticker_dataframe(ticker)
    df=pd.DataFrame(ticker_df.earnings_trend[symbol]['trend'])
    df.replace({'0q':'Current Qtr.','+1q':'Next Qtr.','0y':'Current Year','+1y':'Next Year','+5y':'Next 5 Years (per annum)',
              '-5y':'Past 5 Years (per annum)'},inplace=True)
    df=pd.concat([df,
                df['earningsEstimate'].apply(pd.Series).add_prefix(prefix='earningsEstimate_',axis=1),
                df['revenueEstimate'].apply(pd.Series).add_prefix(prefix='revenueEstimate_',axis=1),
                df['epsTrend'].apply(pd.Series).add_prefix(prefix='epsTrend_',axis=1),
                df['epsRevisions'].apply(pd.Series).add_prefix(prefix='epsRevisions_',axis=1)],axis=1).drop(['epsRevisions','epsTrend','revenueEstimate','earningsEstimate','maxAge'],axis=1)
    df['endDate']=df[['period','endDate']].apply(lambda x:pd.to_datetime(x['endDate']).strftime('%b %Y') if x['period'] in ['Current Qtr.','Next Qtr.'] and x['endDate']!=None
                                                            else pd.to_datetime(x['endDate']).strftime('%Y') if x['period'] not in ['Current Qtr.','Next Qtr.'] and x['endDate']!=None else np.nan,axis=1)
    df.replace(to_replace=[{}],value=np.nan,inplace=True)
    return df

@st.cache_resource
def summary_func(ticker):
    ticker_df=ticker_dataframe(ticker)
    df=ticker_df.summary_detail
    return df


# creating the web layout
st.set_page_config(layout='wide')


with st.container():
    st.title(':green[Finance Portal]')
    st.divider()
    ticker = st.text_input(label='Enter a ticker or multiple comma separated tickers (maximum 5)')
    ticker=ticker.split(',')
    ticker=[x.strip() for x in ticker]
    if ticker[0]!='':
        if len(ticker)>0 and len(ticker)<=5:
            # ticker_df=Ticker(ticker)
            # annual and quarter balance sheet data for the tickers selected
            balance_sheet_annual=balance_sheet_annual_func(ticker) 
            balance_sheet_quarter=balance_sheet_quarter_func(ticker)
            # annual and quarter income statement data for the tickers selected
            income_statement_annual=income_statement_annual_func(ticker)
            income_statement_quarter=income_statement_quarter_func(ticker)
            # annual and quarter cash flow data for the tickers selected
            cash_flow_annual=cash_flow_annual_func(ticker)
            cash_flow_quarter=cash_flow_quarter_func(ticker)
            # historical stock price data for the tickers selected
            stock_price_history=stock_price_history_func(ticker)
            # ticker profiles for the selected tickers
            profile=profile_func(ticker)           
            with st.container():
                col=st.columns(spec=len(ticker))
                for i,j in zip(ticker,range(len(ticker))):
                    with col[j].container():
                        metric_calc=stock_price_history.loc[stock_price_history.symbol==i.upper()].tail(2)[['date','close']].set_index('date').T
                        metric_calc['change']=metric_calc.iloc[:,1]-metric_calc.iloc[:,0]
                        metric_calc['pct_change']=metric_calc.change/metric_calc.iloc[:,0]
                        st.metric(label=i.upper(),value=round(metric_calc.iloc[:,1].close,2),delta=str(round((metric_calc['pct_change'].close)*100,2))+'%')
            with st.container():
                tab=st.tabs(ticker)
                for i,j in zip(ticker,range(len(ticker))):
                    with tab[j]:
                        tab1,tab2,tab3,tab4,tab8=st.tabs(['Profile','Summary','Chart','Financials','Analysis'])
                        with tab1:
                            with st.container():
                                col1,col2=st.columns([0.2,0.4])
                                with col1.container():
                                    st.markdown('###### :green[Address]')
                                    st.markdown('###### :white['+profile[i]['address1'] +']')
                                    st.markdown('###### :white['+profile[i]['city']+', '+profile[i]['state']+' '+profile[i]['zip'] +']')
                                    st.markdown('###### :white['+profile[i]['country'] +']')
                                    st.markdown('###### :green[Phone]')
                                    phone_no=profile[i]['phone'] if 'phone' in profile[i].keys() else 'Not available'
                                    st.markdown('###### :white['+phone_no+']')
                                    st.markdown('###### :green[Website]')
                                    website_=profile[i]['website'] if 'website' in profile[i].keys() else 'Not available'
                                    st.markdown('###### :white['+website_ +']')
                                with col2.container():
                                    st.markdown('###### :green[Sector(s)]'+' :white['+profile[i]['sector'] +']')
                                    st.markdown('###### :green[Industry]'+' :white['+profile[i]['industry'] +']')
                                    st.markdown('###### :green[Full Time Employees]'+' :white['+str(profile[i]['fullTimeEmployees']) +']')
                            st.markdown('#### :blue[Key Executives]')
                            st.data_editor(pd.DataFrame(profile[i]['companyOfficers'])[['name','age','title','yearBorn','fiscalYear',
                                                                                      'totalPay']].rename(columns={'name':'Name','age':'Age','title':'Title',
                                                                                                                   'yearBorn':'Year Born','fiscalYear':'Fiscal Year',
                                                                                                                   'totalPay':'Total Pay'}),
                                            column_config={'Year Born':st.column_config.NumberColumn('Year Born',format='%d'),
                                                           'Fiscal Year':st.column_config.NumberColumn('Fiscal Year',format='%d'),
                                                           'Total Pay':st.column_config.NumberColumn('Total Pay',format='$%d')},
                                                           use_container_width=True,hide_index=True)
                            lat_lon=zip_code.loc[(zip_code['postal code']==str(profile[i]['zip'])) & (zip_code['admin code1']==profile[i]['state']),
                                                 ['latitude','longitude']]
                            lat_lon['point']=1
                            st.map(data=lat_lon,latitude='latitude',longitude='longitude',size=100)
                            st.markdown('#### :blue[Description]')
                            st.write(profile[i]['longBusinessSummary'])
                            st.markdown('#### :blue[Corporate Governance]')
                            st.write('''ISS Governance QualityScore as of ''' +pd.to_datetime(profile[i]['governanceEpochDate']).strftime('%B %d, %Y') +'''.
                                      The pillar scores are Audit: '''+ str(profile[i]['auditRisk'])+'''; Board: '''+str(profile[i]['boardRisk'])+ '''; 
                                     Shareholder Rights: '''+str(profile[i]['shareHolderRightsRisk'])+'''; Compensation: '''+str(profile[i]['compensationRisk'])+'''.
                                     Corporate governance scores courtesy of [Institutional Shareholder Services (ISS)](https://issgovernance.com/quickscore). 
                                     Scores indicate decile rank relative to index or region. A decile score of 1 indicates lower governance risk, while a 10 indicates 
                                     higher governance risk.''')
                        with tab2:
                            with st.container():
                                summary=summary_func(ticker)
                                col1,col2,col3=st.columns(3)
                                with col1.container():
                                    st.markdown('###### :green[Previous Close]  '+str(summary[i]['previousClose']))
                                    st.markdown('---')
                                    st.markdown('###### :green[Open]            '+str(summary[i]['open']))
                                    st.markdown('---')
                                    st.markdown('###### :green[Bid]             '+str(summary[i]['bid']))
                                    st.markdown('---')
                                    st.markdown('###### :green[Ask]             '+str(summary[i]['ask']))
                                    st.markdown('---')
                                    st.markdown('###### :green[Days Range]      '+str(summary[i]['dayLow'])+'-'+str(summary[i]['dayHigh']))
                                    # st.markdown('---')
                                with col2.container():
                                    st.markdown('###### :green[52 Week Range]   '+str(summary[i]['fiftyTwoWeekLow'])+'-'+str(summary[i]['fiftyTwoWeekHigh']))
                                    st.markdown('---')
                                    st.markdown('###### :green[Volume]          '+str(summary[i]['volume']))
                                    st.markdown('---')
                                    st.markdown('###### :green[Avg. Volume]     '+str(summary[i]['averageVolume']))
                                    st.markdown('---')
                                    st.markdown('###### :green[Market Cap]                '+str(summary[i]['marketCap']))
                                    st.markdown('---')
                                    st.markdown('###### :green[Beta (5Y Monthly)]         '+str(round(summary[i]['beta'],2)))
                                    # st.markdown('---')
                                with col3.container():
                                    st.markdown('###### :green[PE Ratio (TTM)]            '+str(round(summary[i]['trailingPE'],2)))
                                    st.markdown('---')
                                    dividend_rate=str(round(summary[i]['dividendRate'],2)) if str(summary[i]['dividendRate'])!='{}' else 'Not Available'
                                    dividend_yield=str(round(summary[i]['dividendYield']*100,2)) if str(summary[i]['dividendYield'])!='{}' else 'Not available'
                                    st.markdown('###### :green[Forward Dividend & Yield]  '+dividend_rate+' ('+dividend_yield+'%)')
                                    st.markdown('---')
                                    ex_dividend_date=str(summary[i]['exDividendDate']) if str(summary[i]['exDividendDate'])!='{}' else 'Not Available'
                                    st.markdown('###### :green[Ex-Dividend Date]          '+ex_dividend_date)
                                    # st.markdown('---')
                        with tab3:
                            col1,col2,col3,col4=st.columns([0.2,0.2,0.2,0.4])
                            with col1.container():
                                selected_tickers=st.multiselect('Select Tickers',options=ticker, default=i)
                            with col2.container():
                                selected_period=st.selectbox('Select Period',options=['Last 30 Days','Last 6 Months','Year to Date',
                                                                                      'Last 1 Year','Last 5 Years', 'All'],index=1,key='0_'+i)
                                if selected_period!='All':
                                    selected_date=stock_price_history.date.max()-datetime.timedelta(days=days(selected_period,stock_price_history.date.max()))
                                else:
                                    selected_date=stock_price_history.date.min()
                            with col3.container():
                                selected_y=st.selectbox('Select Y Axis',options=['Close','Open','High','Low','Volume'],index=0,key=i)
                            fig1=go.Figure()
                            for k in selected_tickers:
                                stock=stock_price_history.loc[(stock_price_history.symbol==k) & (stock_price_history.date>=selected_date)]
                                x=stock.date.values
                                y=stock[selected_y.lower()].values
                                fig1.add_trace(go.Scatter(x=x,y=y,mode='lines+markers',name=k,
                                                          customdata=list(stock[['open','high','low','symbol','volume','close']].to_numpy())))
                                fig1.update_traces(hovertemplate=None,hoverinfo='skip')
                                fig1.update_traces(hovertemplate='<br>Symbol: %{customdata[3]} </br><br>Date: %{x: %Y-%m-%d} </br><br>Open: %{customdata[0]:$.2f}  </br><br>Close: %{customdata[5]:$.2f} </br><br>High: %{customdata[1]:$.2f} </br><br>Low: %{customdata[2]:$.2f} </br><br>Volume: %{customdata[4]}</br><extra></extra>')
                            fig1.update_xaxes(rangeslider_visible=False,
                                              tickformatstops = [
                                                    dict(dtickrange=[86400000, 604800000], value="%e. %b"),
                                                    # dict(dtickrange=[604800000, "M1"], value="%e. %b WK"),
                                                    dict(dtickrange=["M1", "M12"], value="%b '%y"),
                                                    dict(dtickrange=["M12", None], value="%Y")])
                            fig1.update_layout(xaxis=dict(showgrid=False),yaxis=dict(showgrid=False),plot_bgcolor='rgba(15,17,23)',showlegend=True,
                                               hovermode='x unified')
                            with st.container():
                                st.plotly_chart(fig1,use_container_width=True)
                        with tab4:
                            tab5,tab6,tab7=st.tabs(['Balance Sheet','Income Statement','Cash Flow'])
                            with tab5:
                                with st.container():
                                    with st.container():
                                        st.markdown('#### :green['+i.upper()+' Balance Sheet]')
                                    col1,col2,col3=st.columns(3)
                                    with col1.container():
                                        bs_frequency=st.selectbox(label='Frequency',options=['Annual','Quarter'],key=i+'bs_frequency')
                                    with col2.container():
                                        balancesheet_lod=st.selectbox(label='Level of Detail',options=['Short','Full'],key=i+'bs_lod')
                                    if bs_frequency=='Annual':
                                        bs=balance_sheet_annual.loc[balance_sheet_annual.symbol==i]
                                    else:
                                        bs=balance_sheet_quarter.loc[balance_sheet_quarter.symbol==i]
                                    bs['asOfDate']=pd.to_datetime(bs['asOfDate']).dt.strftime('%Y-%m-%d')
                                    bs_full=bs[['TotalAssets','CurrentAssets','CashCashEquivalentsAndShortTermInvestments','CashAndCashEquivalents','CashFinancial','CashEquivalents',
                                            'OtherShortTermInvestments','Receivables','AccountsReceivable','OtherReceivables','Inventory','OtherCurrentAssets','TotalNonCurrentAssets',
                                            'NetPPE','GrossPPE','Properties','LandAndImprovements','MachineryFurnitureEquipment','Leases','AccumulatedDepreciation','InvestmentsAndAdvances',
                                            'InvestmentinFinancialAssets','AvailableForSaleSecurities','OtherInvestments','OtherNonCurrentAssets','TotalLiabilitiesNetMinorityInterest',
                                            'CurrentLiabilities','PayablesAndAccruedExpenses','Payables','AccountsPayable','CurrentDebtAndCapitalLeaseObligation','CurrentDebt',
                                            'CommercialPaper','OtherCurrentBorrowings','CurrentDeferredLiabilities','CurrentDeferredRevenue','OtherCurrentLiabilities',
                                            'TotalNonCurrentLiabilitiesNetMinorityInterest','LongTermDebtAndCapitalLeaseObligation','LongTermDebt','TradeandOtherPayablesNonCurrent',
                                            'OtherNonCurrentLiabilities','TotalEquityGrossMinorityInterest','StockholdersEquity','CapitalStock','CommonStock','RetainedEarnings',
                                            'GainsLossesNotAffectingRetainedEarnings','TotalCapitalization','CommonStockEquity','NetTangibleAssets','WorkingCapital',
                                            'InvestedCapital','TangibleBookValue','TotalDebt','NetDebt','ShareIssued','OrdinarySharesNumber','asOfDate']].set_index('asOfDate').T.reset_index().rename(columns={'index':'Metric'})
                                    bs_full['Metric']=bs_full.Metric.apply(lambda x: sentence_case(x))
                                    bs_short=bs[['asOfDate','TotalAssets','TotalLiabilitiesNetMinorityInterest','TotalEquityGrossMinorityInterest','TotalCapitalization',
                                            'CommonStockEquity','NetTangibleAssets','WorkingCapital','InvestedCapital','TangibleBookValue','TotalDebt','NetDebt',
                                            'ShareIssued','OrdinarySharesNumber']].set_index('asOfDate').T.reset_index().rename(columns={'index':'Metric'})
                                    bs_short['Metric']=bs_short.Metric.apply(lambda x: sentence_case(x))
                                    if balancesheet_lod=='Full':
                                        bs_lod=bs_full
                                    else:
                                        bs_lod=bs_short
                                    with st.container():
                                        st.data_editor(bs_lod,use_container_width=True,hide_index=True)   
                                    def convert_df(df):
                                        return df.to_csv(index=False).encode('utf-8') 
                                    with col3.container():
                                        st.markdown(' ')
                                        st.markdown(' ')
                                        st.download_button(label='Export',data=convert_df(bs_lod),file_name=i+' Balance Sheet.csv') 
                                    with st.container():
                                        st.markdown('#### :green[Balance Sheet Comparison for Selected Tickers and Year]')
                                    col4,col5,col6=st.columns(3)
                                    with col4.container():
                                        bs_selected_tickers=st.multiselect('Select Tickers',options=ticker, default=i,key=i+'bs')
                                    with col5.container():
                                        key=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                        bs_selected_year=st.selectbox('Select Year',options=balance_sheet_annual.loc[balance_sheet_annual.periodType!='TTM'].asOfDate.dt.year.unique(),index=3,key=key)
                                    with st.container():
                                        bs=balance_sheet_annual[['TotalAssets','CurrentAssets','CashCashEquivalentsAndShortTermInvestments','CashAndCashEquivalents','CashFinancial','CashEquivalents',
                                                                'OtherShortTermInvestments','Receivables','AccountsReceivable','OtherReceivables','Inventory','OtherCurrentAssets','TotalNonCurrentAssets',
                                                                'NetPPE','GrossPPE','Properties','LandAndImprovements','MachineryFurnitureEquipment','Leases','AccumulatedDepreciation','InvestmentsAndAdvances',
                                                                'InvestmentinFinancialAssets','AvailableForSaleSecurities','OtherInvestments','OtherNonCurrentAssets','TotalLiabilitiesNetMinorityInterest',
                                                                'CurrentLiabilities','PayablesAndAccruedExpenses','Payables','AccountsPayable','CurrentDebtAndCapitalLeaseObligation','CurrentDebt',
                                                                'CommercialPaper','OtherCurrentBorrowings','CurrentDeferredLiabilities','CurrentDeferredRevenue','OtherCurrentLiabilities',
                                                                'TotalNonCurrentLiabilitiesNetMinorityInterest','LongTermDebtAndCapitalLeaseObligation','LongTermDebt','TradeandOtherPayablesNonCurrent',
                                                                'OtherNonCurrentLiabilities','TotalEquityGrossMinorityInterest','StockholdersEquity','CapitalStock','CommonStock','RetainedEarnings',
                                                                'GainsLossesNotAffectingRetainedEarnings','TotalCapitalization','CommonStockEquity','NetTangibleAssets','WorkingCapital',
                                                                'InvestedCapital','TangibleBookValue','TotalDebt','NetDebt','ShareIssued','OrdinarySharesNumber','symbol','asOfDate','periodType']].copy()
                                        bs['asOfDate']=bs['asOfDate'].dt.strftime('%Y')
                                        bs['index']=bs['symbol']+' - '+bs['asOfDate']
                                        bs_comparison=bs.loc[(bs.symbol.isin(bs_selected_tickers)) & (bs.asOfDate==str(bs_selected_year)) 
                                                             & (bs.periodType!='TTM')].drop(['symbol','asOfDate','periodType'],axis=1).set_index('index').T.reset_index().rename(columns={'index':'Metric'})
                                        key2=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                        bs_comparison['Metric']=bs_comparison.Metric.apply(lambda x: sentence_case(x))
                                        st.data_editor(bs_comparison,use_container_width=True,hide_index=True,key=key2)
                                    with col6.container():
                                        st.markdown(' ')
                                        st.markdown(' ')
                                        st.download_button(label='Export',data=convert_df(bs_comparison),file_name=str(bs_selected_year)+' '+str(bs_selected_tickers)+' Balance Sheet.csv')                         
                            with tab6:
                                with st.container():
                                    with st.container():
                                        st.markdown('#### :green['+i.upper()+' Income Statement]')
                                    col1,col2,col3=st.columns(3)
                                    with col1.container():
                                        is_frequency=st.selectbox(label='Frequency',options=['Annual','Quarter'],key=i+'is_frequency')
                                    with col2.container():
                                        incomestatement_lod=st.selectbox(label='Level of Detail',options=['Short','Full'],key=i+'is_lod')
                                    if is_frequency=='Annual':
                                        is_=income_statement_annual.loc[(income_statement_annual.symbol==i) & (income_statement_annual.periodType!='TTM')]
                                    else:
                                        is_=income_statement_quarter.loc[(income_statement_quarter.symbol==i) & (income_statement_annual.periodType!='TTM')]
                                    is_['asOfDate']=pd.to_datetime(is_['asOfDate']).dt.strftime('%Y-%m-%d')
                                    is_full=is_[['TotalRevenue','OperatingRevenue','CostOfRevenue','GrossProfit','OperatingExpense','SellingGeneralAndAdministration',
                                                'ResearchAndDevelopment','OperatingIncome','NetNonOperatingInterestIncomeExpense','InterestIncomeNonOperating',
                                                'InterestExpenseNonOperating','OtherIncomeExpense','OtherNonOperatingIncomeExpenses','PretaxIncome',
                                                'TaxProvision','NetIncomeCommonStockholders','NetIncome','NetIncomeIncludingNoncontrollingInterests',
                                                'NetIncomeContinuousOperations','DilutedNIAvailtoComStockholders','TotalOperatingIncomeAsReported','TotalExpenses',
                                                'NetIncomeFromContinuingAndDiscontinuedOperation','NormalizedIncome','NetInterestIncome','EBIT', 'EBITDA',
                                                'ReconciledCostOfRevenue','ReconciledDepreciation','NetIncomeFromContinuingOperationNetMinorityInterest',
                                                'NormalizedEBITDA','TaxRateForCalcs','TaxEffectOfUnusualItems','asOfDate']].set_index('asOfDate').T.reset_index().rename(columns={'index':'Metric'})
                                    is_full['Metric']=is_full.Metric.apply(lambda x: sentence_case(x))
                                    is_short=is_[['TotalRevenue','CostOfRevenue','GrossProfit','OperatingExpense','OperatingIncome','NetNonOperatingInterestIncomeExpense',
                                                'OtherIncomeExpense','PretaxIncome','TaxProvision','NetIncomeCommonStockholders','DilutedNIAvailtoComStockholders',
                                                'TotalOperatingIncomeAsReported','TotalExpenses','NetIncomeFromContinuingAndDiscontinuedOperation','NormalizedIncome',
                                                'NetInterestIncome','EBIT', 'EBITDA','ReconciledCostOfRevenue','ReconciledDepreciation','NetIncomeFromContinuingOperationNetMinorityInterest',
                                                'NormalizedEBITDA','TaxRateForCalcs','TaxEffectOfUnusualItems','asOfDate']].set_index('asOfDate').T.reset_index().rename(columns={'index':'Metric'})
                                    is_short['Metric']=is_short.Metric.apply(lambda x: sentence_case(x))
                                    if incomestatement_lod=='Full':
                                        is_lod=is_full
                                    else:
                                        is_lod=is_short
                                    with st.container():
                                        st.data_editor(is_lod,use_container_width=True,hide_index=True) 
                                    def convert_df(df):
                                        return df.to_csv(index=False).encode('utf-8')
                                    with col3.container():
                                        st.markdown(' ')
                                        st.markdown(' ')
                                        st.download_button(label='Export',data=convert_df(is_lod),file_name=i+' Income Statement.csv') 
                                    with st.container():
                                        st.markdown('#### :green[Income Statement Comparison for Selected Tickers and Year]')
                                    col4,col5,col6=st.columns(3)
                                    with col4.container():
                                        is_selected_tickers=st.multiselect('Select Tickers',options=ticker, default=i,key=i+'is')
                                    with col5.container():
                                        key=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                        is_selected_year=st.selectbox('Select Year',options=income_statement_annual.loc[income_statement_annual.periodType!='TTM'].asOfDate.dt.year.unique(),index=3,key=key)
                                    with st.container():
                                        is__=income_statement_annual[['TotalRevenue','OperatingRevenue','CostOfRevenue','GrossProfit','OperatingExpense','SellingGeneralAndAdministration',
                                                                    'ResearchAndDevelopment','OperatingIncome','NetNonOperatingInterestIncomeExpense','InterestIncomeNonOperating',
                                                                    'InterestExpenseNonOperating','OtherIncomeExpense','OtherNonOperatingIncomeExpenses','PretaxIncome',
                                                                    'TaxProvision','NetIncomeCommonStockholders','NetIncome','NetIncomeIncludingNoncontrollingInterests',
                                                                    'NetIncomeContinuousOperations','DilutedNIAvailtoComStockholders','TotalOperatingIncomeAsReported','TotalExpenses',
                                                                    'NetIncomeFromContinuingAndDiscontinuedOperation','NormalizedIncome','NetInterestIncome','EBIT', 'EBITDA',
                                                                    'ReconciledCostOfRevenue','ReconciledDepreciation','NetIncomeFromContinuingOperationNetMinorityInterest',
                                                                    'NormalizedEBITDA','TaxRateForCalcs','TaxEffectOfUnusualItems','symbol','asOfDate','periodType']].copy()
                                        is__['asOfDate']=is__['asOfDate'].dt.strftime('%Y')
                                        is__['index']=is__['symbol']+' - '+is__['asOfDate']
                                        is__comparison=is__.loc[(is__.symbol.isin(is_selected_tickers)) & (is__.asOfDate==str(is_selected_year)) 
                                                             & (is__.periodType!='TTM')].drop(['symbol','asOfDate','periodType'],axis=1).set_index('index').T.reset_index().rename(columns={'index':'Metric'})
                                        key2=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                        is__comparison['Metric']=is__comparison.Metric.apply(lambda x: sentence_case(x))
                                        st.data_editor(is__comparison,use_container_width=True,hide_index=True,key=key2)
                                    with col6.container():
                                        st.markdown(' ')
                                        st.markdown(' ')
                                        st.download_button(label='Export',data=convert_df(is__comparison),file_name=str(is_selected_year)+' '+str(is_selected_tickers)+' Income Statement.csv')  
                            with tab7:
                                with st.container():
                                    with st.container():
                                        st.markdown('#### :green['+i.upper()+' Cash Flow]')
                                    col1,col2,col3=st.columns(3)
                                    with col1.container():
                                        cf_frequency=st.selectbox(label='Frequency',options=['Annual','Quarter'],key=i+'cf_frequency')
                                    with col2.container():
                                        cashflow_lod=st.selectbox(label='Level of Detail',options=['Short','Full'],key=i+'cf_lod')
                                    if cf_frequency=='Annual':
                                        cf=cash_flow_annual.loc[(cash_flow_annual.symbol==i) & (cash_flow_annual.periodType!='TTM')]
                                    else:
                                        cf=cash_flow_quarter.loc[(cash_flow_quarter.symbol==i) & (cash_flow_quarter.periodType!='TTM')]
                                    cf['asOfDate']=pd.to_datetime(cf['asOfDate']).dt.strftime('%Y-%m-%d')
                                    cf_full=cf[['OperatingCashFlow','CashFlowFromContinuingOperatingActivities','NetIncomeFromContinuingOperations','DepreciationAndAmortization',
                                                'DeferredTax','DeferredIncomeTax','StockBasedCompensation','OtherNonCashItems','ChangeInWorkingCapital','ChangeInReceivables',
                                                'ChangesInAccountReceivables','ChangeInInventory','ChangeInPayablesAndAccruedExpense','ChangeInPayable','ChangeInAccountPayable',
                                                'ChangeInOtherCurrentAssets','ChangeInOtherCurrentLiabilities','ChangeInOtherWorkingCapital','InvestingCashFlow','CashFlowFromContinuingInvestingActivities',
                                                'NetPPEPurchaseAndSale','PurchaseOfPPE','NetBusinessPurchaseAndSale','PurchaseOfBusiness','NetInvestmentPurchaseAndSale','PurchaseOfInvestment',
                                                'SaleOfInvestment','NetOtherInvestingChanges','FinancingCashFlow','CashFlowFromContinuingFinancingActivities','NetIssuancePaymentsOfDebt',
                                                'NetLongTermDebtIssuance','LongTermDebtIssuance', 'LongTermDebtPayments','NetCommonStockIssuance','CommonStockIssuance', 'CommonStockPayments',
                                                'CashDividendsPaid','CommonStockDividendPaid','NetOtherFinancingCharges','EndCashPosition','ChangesInCash','BeginningCashPosition',
                                                'IncomeTaxPaidSupplementalData','InterestPaidSupplementalData','CapitalExpenditure','IssuanceOfCapitalStock', 'IssuanceOfDebt',
                                                'RepaymentOfDebt', 'RepurchaseOfCapitalStock','FreeCashFlow','asOfDate']].set_index('asOfDate').T.reset_index().rename(columns={'index':'Metric'})
                                    cf_full['Metric']=cf_full.Metric.apply(lambda x: sentence_case(x))
                                    cf_short=cf[['OperatingCashFlow','InvestingCashFlow','FinancingCashFlow','EndCashPosition','IncomeTaxPaidSupplementalData',
                                                    'InterestPaidSupplementalData','CapitalExpenditure','IssuanceOfCapitalStock','IssuanceOfDebt','RepaymentOfDebt',
                                                    'RepurchaseOfCapitalStock','FreeCashFlow','asOfDate']].set_index('asOfDate').T.reset_index().rename(columns={'index':'Metric'})
                                    cf_short['Metric']=cf_short.Metric.apply(lambda x: sentence_case(x))
                                    if cashflow_lod=='Full':
                                        cf_lod=cf_full
                                    else:
                                        cf_lod=cf_short
                                    with st.container():
                                        st.data_editor(cf_lod,use_container_width=True,hide_index=True)
                                    with col3.container():
                                        st.markdown(' ')
                                        st.markdown(' ')
                                        st.download_button(label='Export',data=convert_df(cf_lod),file_name=i+' Cash Flow.csv')
                                    with st.container():
                                        st.markdown('#### :green[Cash Flow Comparison for Selected Tickers and Year]')
                                    col4,col5,col6=st.columns(3)
                                    with col4.container():
                                        cf_selected_tickers=st.multiselect('Select Tickers',options=ticker, default=i,key=i+'cf')
                                    with col5.container():
                                        key=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                        cf_selected_year=st.selectbox('Select Year',options=cash_flow_annual.loc[cash_flow_annual.periodType!='TTM'].asOfDate.dt.year.unique(),index=3,key=key)
                                    with st.container():
                                        cf=cash_flow_annual[['OperatingCashFlow','CashFlowFromContinuingOperatingActivities','NetIncomeFromContinuingOperations','DepreciationAndAmortization',
                                                                'DeferredTax','DeferredIncomeTax','StockBasedCompensation','OtherNonCashItems','ChangeInWorkingCapital','ChangeInReceivables',
                                                                'ChangesInAccountReceivables','ChangeInInventory','ChangeInPayablesAndAccruedExpense','ChangeInPayable','ChangeInAccountPayable',
                                                                'ChangeInOtherCurrentAssets','ChangeInOtherCurrentLiabilities','ChangeInOtherWorkingCapital','InvestingCashFlow','CashFlowFromContinuingInvestingActivities',
                                                                'NetPPEPurchaseAndSale','PurchaseOfPPE','NetBusinessPurchaseAndSale','PurchaseOfBusiness','NetInvestmentPurchaseAndSale','PurchaseOfInvestment',
                                                                'SaleOfInvestment','NetOtherInvestingChanges','FinancingCashFlow','CashFlowFromContinuingFinancingActivities','NetIssuancePaymentsOfDebt',
                                                                'NetLongTermDebtIssuance','LongTermDebtIssuance', 'LongTermDebtPayments','NetCommonStockIssuance','CommonStockIssuance', 'CommonStockPayments',
                                                                'CashDividendsPaid','CommonStockDividendPaid','NetOtherFinancingCharges','EndCashPosition','ChangesInCash','BeginningCashPosition',
                                                                'IncomeTaxPaidSupplementalData','InterestPaidSupplementalData','CapitalExpenditure','IssuanceOfCapitalStock', 'IssuanceOfDebt',
                                                                'RepaymentOfDebt', 'RepurchaseOfCapitalStock','FreeCashFlow','symbol','asOfDate','periodType']].copy()
                                        cf['asOfDate']=cf['asOfDate'].dt.strftime('%Y')
                                        cf['index']=cf['symbol']+' - '+cf['asOfDate']
                                        cf_comparison=cf.loc[(cf.symbol.isin(cf_selected_tickers)) & (cf.asOfDate==str(cf_selected_year)) 
                                                             & (cf.periodType!='TTM')].drop(['symbol','asOfDate','periodType'],axis=1).set_index('index').T.reset_index().rename(columns={'index':'Metric'})
                                        key2=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                        cf_comparison['Metric']=cf_comparison.Metric.apply(lambda x: sentence_case(x))
                                        st.data_editor(cf_comparison,use_container_width=True,hide_index=True,key=key2)
                                    with col6.container():
                                        st.markdown(' ')
                                        st.markdown(' ')
                                        st.download_button(label='Export',data=convert_df(cf_comparison),file_name=str(cf_selected_year)+' '+str(cf_selected_tickers)+' Cash Flow.csv')
                        with tab8:
                            trend_estimates=earning_trend_func(ticker,i)
                            earnings_estimate=trend_estimates[['period','endDate','earningsEstimate_avg',
                                                                'earningsEstimate_low', 'earningsEstimate_high',
                                                                'earningsEstimate_yearAgoEps', 'earningsEstimate_numberOfAnalysts',
                                                                'earningsEstimate_growth']][:4]
                            earnings_estimate.columns=earnings_estimate.columns.str.replace('earningsEstimate_','')
                            earnings_estimate['cols']=earnings_estimate['period']+' ('+earnings_estimate['endDate']+')'
                            earnings_estimate=earnings_estimate.set_index('cols').drop(['period','endDate'],axis=1).T.reset_index().rename(columns={'index':'Earnings Estimate'})
                            revenue_estimate=trend_estimates[['period','endDate','revenueEstimate_avg', 'revenueEstimate_low',
                                                                'revenueEstimate_high', 'revenueEstimate_numberOfAnalysts',
                                                                'revenueEstimate_yearAgoRevenue', 'revenueEstimate_growth']][:4]
                            revenue_estimate.columns=revenue_estimate.columns.str.replace('revenueEstimate_','')
                            revenue_estimate['cols']=revenue_estimate['period']+' ('+revenue_estimate['endDate']+')'
                            revenue_estimate=revenue_estimate.set_index('cols').drop(['period','endDate'],axis=1).T.reset_index().rename(columns={'index':'Revenue Estimate'})
                            eps_revisions=trend_estimates[['period','endDate','epsRevisions_upLast7days',
                                                            'epsRevisions_upLast30days', 'epsRevisions_downLast30days',
                                                            'epsRevisions_downLast90days']][:4]
                            eps_revisions.columns=eps_revisions.columns.str.replace('epsRevisions_','')
                            eps_revisions['cols']=eps_revisions['period']+' ('+eps_revisions['endDate']+')'
                            eps_revisions=eps_revisions.set_index('cols').drop(['period','endDate'],axis=1).T.reset_index().rename(columns={'index':'EPS Revisions'})
                            eps_trend=trend_estimates[['period','endDate','epsTrend_current', 'epsTrend_7daysAgo', 'epsTrend_30daysAgo',
                                                        'epsTrend_60daysAgo', 'epsTrend_90daysAgo']][:4]
                            eps_trend.columns=eps_trend.columns.str.replace('epsTrend_','')
                            eps_trend['cols']=eps_trend['period']+' ('+eps_trend['endDate']+')'
                            eps_trend=eps_trend.set_index('cols').drop(['period','endDate'],axis=1).T.reset_index().rename(columns={'index':'EPS Trend'})
                            with st.container():
                                st.markdown('#### :green[Earnings Estimate]')
                                key3=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                earnings_estimate['Earnings Estimate']=earnings_estimate['Earnings Estimate'].apply(lambda x: sentence_case(x))
                                st.data_editor(earnings_estimate,use_container_width=True,hide_index=True,key=key3)
                                st.markdown('#### :green[Revenue Estimate]')
                                key4=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                revenue_estimate['Revenue Estimate']=revenue_estimate['Revenue Estimate'].apply(lambda x: sentence_case(x))
                                st.data_editor(revenue_estimate,use_container_width=True,hide_index=True,key=key4)
                                st.markdown('#### :green[EPS Trend]')
                                key5=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                eps_trend['EPS Trend']=eps_trend['EPS Trend'].apply(lambda x: sentence_case(x))
                                st.data_editor(eps_trend,use_container_width=True,hide_index=True,key=key5)
                                st.markdown('#### :green[EPS Revisions]')
                                key6=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                eps_revisions['EPS Revisions']=eps_revisions['EPS Revisions'].apply(lambda x: sentence_case(x))
                                st.data_editor(eps_revisions,use_container_width=True,hide_index=True,key=key6)
                                st.markdown('#### :green[Growth Estimates]')
                                key7=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
                                te=trend_estimates[['period','growth']].rename(columns={'growth':'Growth','period':'Growth Estimates'})
                                st.data_editor(te,use_container_width=True,hide_index=True,key=key7)                       
        else:
            st.markdown('#### :white[You have entered more than 5 tickers]')
    else:
        st.markdown('#### :white[Please enter your ticker(s)]')
