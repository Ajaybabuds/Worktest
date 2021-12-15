import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg
import warnings

warnings.filterwarnings('ignore')
st.title("Welcome to NetScore Analytics")
c_yr = datetime.today().year


def date_clean(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe['Month'] = dataframe['Date'].dt.month
    dataframe['Day'] = dataframe['Date'].dt.day_name()
    dataframe['Quarter'] = dataframe['Date'].dt.quarter
    dataframe['Season'] = dataframe['Date'].dt.month.apply(lambda x: x % 12 // 3 + 1)
    return dataframe


def one_dim(dataframe, x, y, z):
    return dataframe[y].groupby(dataframe[x]).agg(z).reset_index()


def sfr(dataframe):
    for i in dataframe.select_dtypes(include='object').columns:
        dataframe[i] = dataframe[i].str.capitalize()
    return dataframe


def few_data(dataframe, year):
    dataframe = dataframe.loc[dataframe['Year'] == year]
    return dataframe


def mul_grp(dataframe, x, y):
    if dataframe[y].dtype == 'object':
        return dataframe.groupby(x).agg({y: lambda x: len(x.value_counts().index)}).unstack(0, fill_value=0)
    else:
        return dataframe.groupby(x).agg({y: 'sum'}).unstack(0, fill_value=0)


def top_most(dataframe, field, val, grp, met):
    return dataframe.loc[(dataframe[field] == val)].groupby([grp]).agg(
        {met: lambda x: x.describe(include='object')['top']})


def convert(df):
    return df.to_csv().encode('utf-8')


def time_series(dataframe, predict=30):
    x = dataframe[dataframe.columns[0]]
    model = AutoReg(x, lags=200)
    fit = model.fit()
    result = []
    for i in range(len(x), len(x) + predict):
        yhat = fit.predict(i, i)
        result.append(yhat)
    new_result = [result[i][0] for i in range(len(result))]
    sr = pd.DataFrame(new_result, index=pd.date_range(dataframe.index[-1], periods=predict + 1)[1:])
    sr = sr.rename(columns={0: dataframe.columns[0]})
    return sr


def rfm(dataframe, col, col1, col_value, y):
    dataframe = dataframe.loc[dataframe['Date'] >= y]
    dataframe = dataframe.loc[dataframe[col1] == col_value]
    rec = dataframe.groupby(by=col, as_index=False)['Date'].max()
    recent_date = rec['Date'].max()
    rec['recency'] = rec['Date'].apply(lambda x: (recent_date - x).days)
    freq = dataframe.groupby(by=col, as_index=False).agg({'Date': 'count', 'Amount': 'sum'})
    freq.columns = [col, 'Frequency', 'Monetary']
    rfm = rec.merge(freq, on=col)
    rfm = rfm.drop(['Date'], axis=1)
    rfm['R_rank'] = rfm['recency'].rank(ascending=False)
    rfm['F_rank'] = rfm['Frequency'].rank(ascending=True)
    rfm['M_rank'] = rfm['Monetary'].rank(ascending=True)
    rfm['R_norm'] = (rfm['R_rank'] / rfm['R_rank'].max()) * 100
    rfm['F_norm'] = (rfm['F_rank'] / rfm['F_rank'].max()) * 100
    rfm['M_norm'] = (rfm['M_rank'] / rfm['M_rank'].max()) * 100
    rfm['RFM score'] = 0.15 * rfm['R_norm'] + 0.28 * rfm['F_norm'] + 0.57 * rfm['M_norm']
    rfm['Rating'] = np.where(rfm['RFM score'] > 75, "High",
                             (np.where(rfm['RFM score'] < 50, "Low", 'Medium')))

    return rfm.sort_values(by='Rating')


data = pd.read_csv(r"https://raw.githubusercontent.com/Ajaybabuds/Worktest/main/Store.csv", quotechar='"', encoding='utf-8', low_memory=False)
data = data.iloc[:, 1:]
data = data.loc[data['Customer'].str.contains('Test') == False]
data = sfr(date_clean(data))
data = data.loc[data['Amount'] > 0]
report = st.sidebar.selectbox("Select a Reports", options=['Sale Report', 'Customer Report', 'Item Report', 'Compare',
                                                           "Multi-dimensional", 'Top Most',
                                                           'Forecast',
                                                           'RFM'])
if 'Sale Report' in report:
    st.write("Choose a Results according to below")
    sales = st.sidebar.selectbox("Select Field", options=['None', 'Current Year', 'Previous Year', 'Custom Range'])
    mnth = st.sidebar.selectbox("Select a field", options=['Month', 'Quarter', 'Season'])
    var = st.sidebar.selectbox("Select a filter", options=data.select_dtypes(include=['float64']).columns)
    cal = st.sidebar.selectbox("Select a metric", options=['sum', 'mean', 'max', 'min', 'count'])
    total = st.sidebar.selectbox("Select Total", options=['Total', 'Difference', 'Growth Rate'])
    if "None" in sales:
        st.write("Please Select Field!!")
    elif 'Current Year' in sales:
        if "Total" in total:
            st.write("Current Year" + " " + total + " " + cal + " " + var + " " + "by" + " " + mnth)
            tot = one_dim(few_data(data, c_yr), mnth, var, cal).set_index([mnth])
            if 'Season' in mnth:
                idx = ['Winter', 'Spring', 'Summer', 'Fall']
                tot.index = [idx[i - 1] for i in tot.index.values]
                st.table(tot.style.hide_index())
                st.download_button("Download File", data=convert(tot), file_name="report.csv")
            elif 'Quarter' in mnth:
                idx = ['Q1', 'Q2', 'Q3', 'Q4']
                tot.index = [idx[i - 1] for i in tot.index.values]
                st.table(tot)
                hide_st_style = """
                            <style>
                            #MainMenu {visibility: hidden;}
                            footer {visibility: hidden;}
                            header {visibility: hidden;}
                            </style>
                            """
                st.markdown(hide_st_style, unsafe_allow_html=True)
            else:
                idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                tot.index = [idx[i - 1] for i in tot.index.values]
                st.table(tot)
        elif "Difference" in total:
            pct = one_dim(few_data(data, c_yr), mnth, var, cal).set_index([mnth]).diff().dropna()
            if 'Season' in mnth:
                idx = ['Winter', 'Spring', 'Summer', 'Fall']
                pct.index = [idx[i - 1] for i in pct.index.values]
                st.dataframe(pct)
            elif 'Quarter' in mnth:
                idx = ['Q1', 'Q2', 'Q3', 'Q4']
                pct.index = [idx[i - 1] for i in pct.index.values]
                st.table(pct)
            else:
                idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                pct.index = [idx[i - 1] for i in pct.index.values]
                st.table(pct)
        else:
            dif = pd.DataFrame(one_dim(few_data(data, c_yr), mnth, var, cal)).set_index([mnth]).pct_change().dropna()
            if 'Season' in mnth:
                idx = ['Winter', 'Spring', 'Summer', 'Fall']
                dif.index = [idx[i - 1] for i in dif.index.values]
                st.table(dif)
            elif 'Quarter' in mnth:
                idx = ['Q1', 'Q2', 'Q3', 'Q4']
                dif.index = [idx[i - 1] for i in dif.index.values]
                st.table(dif)
            else:
                idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                dif.index = [idx[i - 1] for i in dif.index.values]
                st.table(dif)
    elif "Previous Year" in sales:
        if "Total" in total:
            st.write("Current Year" + " " + total + " " + cal + " " + var + " " + "by" + " " + mnth)
            tot = one_dim(few_data(data, c_yr - 1), mnth, var, cal).set_index([mnth])
            if 'Season' in mnth:
                idx = ['Winter', 'Spring', 'Summer', 'Fall']
                tot.index = [idx[i - 1] for i in tot.index.values]
                st.table(tot)
            elif 'Quarter' in mnth:
                idx = ['Q1', 'Q2', 'Q3', 'Q4']
                tot.index = [idx[i - 1] for i in tot.index.values]
                st.table(tot)
            else:
                idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                tot.index = [idx[i - 1] for i in tot.index.values]
                st.table(tot)
        elif "Difference" in total:
            pct = one_dim(few_data(data, c_yr - 1), mnth, var, cal).set_index([mnth]).diff().dropna()
            if 'Season' in mnth:
                idx = ['Winter', 'Spring', 'Summer', 'Fall']
                pct.index = [idx[i - 1] for i in pct.index.values]
                st.table(pct)
            elif 'Quarter' in mnth:
                idx = ['Q1', 'Q2', 'Q3', 'Q4']
                pct.index = [idx[i - 1] for i in pct.index.values]
                st.table(pct)
            else:
                idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                pct.index = [idx[i - 1] for i in pct.index.values]
                st.dataframe(pct)
        else:
            dif = pd.DataFrame(one_dim(few_data(data, c_yr - 1), mnth, var, cal)).set_index(
                [mnth]).pct_change().dropna()
            if 'Season' in mnth:
                idx = ['Winter', 'Spring', 'Summer', 'Fall']
                dif.index = [idx[i - 1] for i in dif.index.values]
                st.table(dif)
            elif 'Quarter' in mnth:
                idx = ['Q1', 'Q2', 'Q3', 'Q4']
                dif.index = [idx[i - 1] for i in dif.index.values]
                st.table(dif)
            else:
                idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                dif.index = [idx[i - 1] for i in dif.index.values]
                st.table(dif)

    else:
        dt1, dt2 = st.columns(2)
        with dt1:
            s1 = st.date_input("From")
            s1 = pd.to_datetime(s1)
        with dt2:
            s2 = st.date_input("To")
            s2 = pd.to_datetime(s2)
        new_data = data.loc[(data['Date'] >= s1) & (data['Date'] <= s2)]
        if "Total" in total:
            st.write("Current Year" + " " + total + " " + cal + " " + var + " " + "by" + " " + mnth)
            tot = one_dim(new_data, mnth, var, cal).set_index([mnth])
            if 'Season' in mnth:
                idx = ['Winter', 'Spring', 'Summer', 'Fall']
                tot.index = [idx[i - 1] for i in tot.index.values]
                st.table(tot)
            elif 'Quarter' in mnth:
                idx = ['Q1', 'Q2', 'Q3', 'Q4']
                tot.index = [idx[i - 1] for i in tot.index.values]
                st.table(tot)
            else:
                idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                tot.index = [idx[i - 1] for i in tot.index.values]
                st.table(tot)
        elif "Difference" in total:
            pct = one_dim(new_data, mnth, var, cal).set_index([mnth]).diff().dropna()
            if 'Season' in mnth:
                idx = ['Winter', 'Spring', 'Summer', 'Fall']
                pct.index = [idx[i - 1] for i in pct.index.values]
                st.table(pct)
            elif 'Quarter' in mnth:
                idx = ['Q1', 'Q2', 'Q3', 'Q4']
                pct.index = [idx[i - 1] for i in pct.index.values]
                st.table(pct)
            else:
                idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                pct.index = [idx[i - 1] for i in pct.index.values]
                st.table(pct)
        else:
            dif = pd.DataFrame(one_dim(new_data, mnth, var, cal)).set_index([mnth]).pct_change().dropna()
            if 'Season' in mnth:
                idx = ['Winter', 'Spring', 'Summer', 'Fall']
                dif.index = [idx[i - 1] for i in dif.index.values]
                st.table(dif.style.format({var: '{:,.2%}'.format}))
            elif 'Quarter' in mnth:
                idx = ['Q1', 'Q2', 'Q3', 'Q4']
                dif.index = [idx[i - 1] for i in dif.index.values]
                st.table(dif.style.format({var: '{:,.2%}'.format}))
            else:
                idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                dif.index = [idx[i - 1] for i in dif.index.values]
                st.table(dif.style.format({var: '{:,.2%}'.format}))
elif "Customer Report" in report:
    cust = st.sidebar.selectbox("Select the Customer", options=data['Customer'].value_counts().index)
    for i in data['Customer'].value_counts().index:
        if i in cust:
            cr = data.loc[data['Customer'] == i]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                mnth = st.sidebar.selectbox("Select a field",
                                            options=data.select_dtypes(exclude=['datetime64[ns]']).columns)
            with col2:
                var = st.sidebar.selectbox("Select a filter", options=data.select_dtypes(include=['float64']).columns)
            with col3:
                cal = st.sidebar.selectbox("Select a metric", options=['sum', 'mean', 'max', 'min', 'count'])
            with col4:
                total = st.sidebar.selectbox("Select Total", options=['Total', 'Difference', 'Growth Rate'])
            if "Total" in total:
                tot = one_dim(cr, mnth, var, cal).set_index([mnth])
                if 'Season' in mnth:
                    idx = ['Winter', 'Spring', 'Summer', 'Fall']
                    tot.index = [idx[i - 1] for i in tot.index.values]
                    st.table(tot)
                elif 'Quarter' in mnth:
                    idx = ['Q1', 'Q2', 'Q3', 'Q4']
                    tot.index = [idx[i - 1] for i in tot.index.values]
                    st.table(tot)
                elif 'Month' in mnth:
                    idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    tot.index = [idx[i - 1] for i in tot.index.values]
                    st.table(tot)
                else:
                    st.table(tot)
            elif "Difference" in total:
                st.write("Current Year" + " " + total + " " + var + " " + "by" + " " + mnth)
                pct = one_dim(cr, mnth, var, cal).set_index([mnth]).diff().dropna()
                if 'Season' in mnth:
                    idx = ['Winter', 'Spring', 'Summer', 'Fall']
                    pct.index = [idx[i - 1] for i in pct.index.values]
                    st.table(pct)
                elif 'Quarter' in mnth:
                    idx = ['Q1', 'Q2', 'Q3', 'Q4']
                    pct.index = [idx[i - 1] for i in pct.index.values]
                    st.table(pct)
                elif 'MOnth' in mnth:
                    idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    pct.index = [idx[i - 1] for i in pct.index.values]
                    st.table(pct)
                else:
                    st.table(pct)
            else:
                dif = pd.DataFrame(one_dim(cr, mnth, var, cal)).set_index([mnth]).pct_change().dropna()
                if 'Season' in mnth:
                    idx = ['Winter', 'Spring', 'Summer', 'Fall']
                    dif.index = [idx[i - 1] for i in dif.index.values]
                    st.table(dif.style.format({var: '{:,.2%}'.format}))
                elif 'Quarter' in mnth:
                    idx = ['Q1', 'Q2', 'Q3', 'Q4']
                    dif.index = [idx[i - 1] for i in dif.index.values]
                    st.table(dif.style.format({var: '{:,.2%}'.format}))
                elif "Month" in mnth:
                    idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    dif.index = [idx[i - 1] for i in dif.index.values]
                    st.table(dif.style.format({var: '{:,.2%}'.format}))
                else:
                    st.table(dif.style.format({var: '{:,.2%}'.format}))
elif "Item Report" in report:
    itm = st.sidebar.selectbox("Select item", options=data['Item'].unique())
    yr = st.sidebar.selectbox("Choose a Year", options=['Current year', 'Previous Year', 'Custom Range'])
    if 'Current year' in yr:
        for j in data['Item'].value_counts().index:
            if j in itm:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    mnth = st.sidebar.selectbox("Select a field",
                                                options=data.select_dtypes(exclude=['datetime64[ns]']).columns)
                with col2:
                    var = st.sidebar.selectbox("Select a filter",
                                               options=data.select_dtypes(include=['float64']).columns)
                with col3:
                    cal = st.sidebar.selectbox("Select a metric",
                                               options=['sum', 'mean', 'max', 'min', 'count'])
                with col4:
                    total = st.sidebar.selectbox("Select Total", options=['Total', 'Difference', 'Growth Rate'])
                ir = data.loc[(data['Item'] == j) & (data['Year'] == c_yr)]
                if "Total" in total:
                    st.write(str(c_yr) + " " + total + " " + cal + " " + var + " " + "by" + " " + mnth)
                    tot = one_dim(ir, mnth, var, cal).set_index([mnth])
                    if 'Season' in mnth:
                        idx = ['Winter', 'Spring', 'Summer', 'Fall']
                        tot.index = [idx[i - 1] for i in tot.index.values]
                        st.dataframe(tot)
                    elif 'Quarter' in mnth:
                        idx = ['Q1', 'Q2', 'Q3', 'Q4']
                        tot.index = [idx[i - 1] for i in tot.index.values]
                        st.dataframe(tot)
                    elif 'Month' in mnth:
                        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                               'Dec']
                        tot.index = [idx[i - 1] for i in tot.index.values]
                        st.dataframe(tot)
                    else:
                        st.dataframe(tot)
                elif "Difference" in total:
                    st.write(str(c_yr) + " " + total + " " + var + " " + "by" + " " + mnth)
                    pct = one_dim(ir, mnth, var, cal).set_index([mnth]).diff().dropna()
                    if 'Season' in mnth:
                        idx = ['Winter', 'Spring', 'Summer', 'Fall']
                        pct.index = [idx[i - 1] for i in pct.index.values]
                        st.dataframe(pct)
                    elif 'Quarter' in mnth:
                        idx = ['Q1', 'Q2', 'Q3', 'Q4']
                        pct.index = [idx[i - 1] for i in pct.index.values]
                        st.dataframe(pct)
                    elif 'Month' in mnth:
                        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                               'Dec']
                        pct.index = [idx[i - 1] for i in pct.index.values]
                        st.dataframe(pct)
                    else:
                        st.dataframe(pct)
                else:
                    st.write(str(c_yr) + " " + total + " " + cal + " " + var + " " + "by" + " " + mnth)
                    dif = pd.DataFrame(one_dim(ir, mnth, var, cal)).set_index([mnth]).pct_change().dropna()
                    if 'Season' in mnth:
                        idx = ['Winter', 'Spring', 'Summer', 'Fall']
                        dif.index = [idx[i - 1] for i in dif.index.values]
                        st.table(dif.style.format({var: '{:,.2%}'.format}))
                    elif 'Quarter' in mnth:
                        idx = ['Q1', 'Q2', 'Q3', 'Q4']
                        dif.index = [idx[i - 1] for i in dif.index.values]
                        st.table(dif.style.format({var: '{:,.2%}'.format}))
                    elif "Month" in mnth:
                        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                               'Dec']
                        dif.index = [idx[i - 1] for i in dif.index.values]
                        st.table(dif.style.format({var: '{:,.2%}'.format}))
                    else:
                        st.dataframe(dif)
    elif "Previous Year" in yr:
        for j in data['Item'].value_counts().index:
            if j in itm:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    mnth = st.sidebar.selectbox("Select a field",
                                                options=data.select_dtypes(exclude=['datetime64[ns]']).columns)
                with col2:
                    var = st.sidebar.selectbox("Select a filter",
                                               options=data.select_dtypes(include=['float64']).columns)
                with col3:
                    cal = st.sidebar.selectbox("Select a metric",
                                               options=['sum', 'mean', 'max', 'min', 'count'])
                with col4:
                    total = st.sidebar.selectbox("Select Total", options=['Total', 'Difference', 'Growth Rate'])
                ir = data.loc[(data['Item'] == j) & (data['Year'] == c_yr - 1)]
                if "Total" in total:
                    st.write(str(c_yr - 1) + " " + total + " " + cal + " " + var + " " + "by" + " " + mnth)
                    tot = one_dim(ir, mnth, var, cal).set_index([mnth])
                    if 'Season' in mnth:
                        idx = ['Winter', 'Spring', 'Summer', 'Fall']
                        tot.index = [idx[i - 1] for i in tot.index.values]
                        st.dataframe(tot)
                    elif 'Quarter' in mnth:
                        idx = ['Q1', 'Q2', 'Q3', 'Q4']
                        tot.index = [idx[i - 1] for i in tot.index.values]
                        st.dataframe(tot)
                    elif 'Month' in mnth:
                        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                               'Dec']
                        tot.index = [idx[i - 1] for i in tot.index.values]
                        st.dataframe(tot)
                    else:
                        if st.checkbox("Select Number of Items to be Display"):
                            n = st.number_input("Enter", 5)
                            rad = st.radio("Sort values by Order", ['Top', 'Bottom'])
                            if "Top" in rad:
                                st.dataframe(tot.sort_values(by=var, ascending=False).head(n))
                            else:
                                st.dataframe(tot.sort_values(by=var).head(n))
                elif "Difference" in total:
                    st.write(str(c_yr - 1) + " " + total + " " + var + " " + "by" + " " + mnth)
                    pct = one_dim(ir, mnth, var, cal).set_index([mnth]).diff().dropna()
                    if 'Season' in mnth:
                        idx = ['Winter', 'Spring', 'Summer', 'Fall']
                        pct.index = [idx[i - 1] for i in pct.index.values]
                        st.dataframe(pct)
                    elif 'Quarter' in mnth:
                        idx = ['Q1', 'Q2', 'Q3', 'Q4']
                        pct.index = [idx[i - 1] for i in pct.index.values]
                        st.dataframe(pct)
                    elif 'Month' in mnth:
                        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                               'Dec']
                        pct.index = [idx[i - 1] for i in pct.index.values]
                        st.dataframe(pct)
                    else:
                        if st.checkbox("Select Number of Items to be Display"):
                            n = st.number_input("Total Items Selected is", 5)
                            rad = st.radio("Sort values by Order", ['Top', 'Bottom'])
                            if "Top" in rad:
                                st.dataframe(pct.sort_values(by=var, ascending=False).head(n))
                            else:
                                st.dataframe(pct.sort_values(by=var).head(n))
                else:
                    st.write(str(c_yr - 1) + " " + total + " " + cal + " " + var + " " + "by" + " " + mnth)
                    dif = pd.DataFrame(one_dim(ir, mnth, var, cal)).set_index([mnth]).pct_change().dropna()
                    if 'Season' in mnth:
                        idx = ['Winter', 'Spring', 'Summer', 'Fall']
                        dif.index = [idx[i - 1] for i in dif.index.values]
                        st.table(dif.style.format({var: '{:,.2%}'.format}))
                    elif 'Quarter' in mnth:
                        idx = ['Q1', 'Q2', 'Q3', 'Q4']
                        dif.index = [idx[i - 1] for i in dif.index.values]
                        st.table(dif.style.format({var: '{:,.2%}'.format}))
                    elif "Month" in mnth:
                        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                               'Dec']
                        dif.index = [idx[i - 1] for i in dif.index.values]
                        st.table(dif.style.format({var: '{:,.2%}'.format}))
                    else:
                        if st.checkbox("Select Number of Items to be Display"):
                            n = st.number_input("Enter", 5)
                            rad = st.radio("Sort values by Order", ['Top', 'Bottom'])
                            if "Top" in rad:
                                st.dataframe(dif.sort_values(by=var, ascending=False).head(n))
                            else:
                                st.dataframe(dif.sort_values(by=var).head(n))
    else:
        dt1, dt2 = st.columns(2)
        with dt1:
            s1 = st.date_input("From")
            s1 = pd.to_datetime(s1)
        with dt2:
            s2 = st.date_input("To")
            s2 = pd.to_datetime(s2)
        for j in data['Item'].value_counts().index:
            if j in itm:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    mnth = st.sidebar.selectbox("Select a field",
                                                options=data.select_dtypes(exclude=['datetime64[ns]']).columns)
                with col2:
                    var = st.sidebar.selectbox("Select a filter",
                                               options=data.select_dtypes(include=['float64']).columns)
                with col3:
                    cal = st.sidebar.selectbox("Select a metric",
                                               options=['sum', 'mean', 'max', 'min', 'count'])
                with col4:
                    total = st.sidebar.selectbox("Select Total", options=['Total', 'Difference', 'Growth Rate'])
                ir = data.loc[(data['Item'] == j) & (data['Date'] >= s1) & (data['Date'] <= s2)]
                if "Total" in total:
                    tot = one_dim(ir, mnth, var, cal).set_index([mnth])
                    if 'Season' in mnth:
                        idx = ['Winter', 'Spring', 'Summer', 'Fall']
                        tot.index = [idx[i - 1] for i in tot.index.values]
                        st.dataframe(tot)
                    elif 'Quarter' in mnth:
                        idx = ['Q1', 'Q2', 'Q3', 'Q4']
                        tot.index = [idx[i - 1] for i in tot.index.values]
                        st.dataframe(tot)
                    elif 'Month' in mnth:
                        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                               'Dec']
                        tot.index = [idx[i - 1] for i in tot.index.values]
                        st.dataframe(tot)
                    else:
                        if st.checkbox("Select Number of Items to be Display"):
                            n = st.number_input("Enter", 5)
                            rad = st.radio("Sort values by Order", ['Top', 'Bottom'])
                            if "Top" in rad:
                                st.dataframe(tot.head(n).sort_values(by=var, ascending=False))
                            else:
                                st.dataframe(tot.head(n).sort_values(by=var))
                elif "Difference" in total:
                    pct = one_dim(ir, mnth, var, cal).set_index([mnth]).diff().dropna()
                    if 'Season' in mnth:
                        idx = ['Winter', 'Spring', 'Summer', 'Fall']
                        pct.index = [idx[i - 1] for i in pct.index.values]
                        st.dataframe(pct)
                    elif 'Quarter' in mnth:
                        idx = ['Q1', 'Q2', 'Q3', 'Q4']
                        pct.index = [idx[i - 1] for i in pct.index.values]
                        st.dataframe(pct)
                    elif 'Month' in mnth:
                        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                               'Dec']
                        pct.index = [idx[i - 1] for i in pct.index.values]
                        st.dataframe(pct)
                    else:
                        if st.checkbox("Select Number of Items to be Display"):
                            n = st.number_input("Enter", 5)
                            rad = st.radio("Sort values by Order", ['Top', 'Bottom'])
                            if "Top" in rad:
                                st.dataframe(pct.head(n).sort_values(by=var, ascending=False))
                            else:
                                st.dataframe(pct.head(n).sort_values(by=var))
                else:
                    st.write(yr + " " + total + " " + cal + " " + var + " " + "by" + " " + mnth)
                    dif = pd.DataFrame(one_dim(ir, mnth, var, cal)).set_index([mnth]).pct_change().dropna()
                    if 'Season' in mnth:
                        idx = ['Winter', 'Spring', 'Summer', 'Fall']
                        dif.index = [idx[i - 1] for i in dif.index.values]
                        st.table(dif.style.format({var: '{:,.2%}'.format}))
                    elif 'Quarter' in mnth:
                        idx = ['Q1', 'Q2', 'Q3', 'Q4']
                        dif.index = [idx[i - 1] for i in dif.index.values]
                        st.table(dif.style.format({var: '{:,.2%}'.format}))
                    elif "Month" in mnth:
                        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                               'Dec']
                        dif.index = [idx[i - 1] for i in dif.index.values]
                        st.table(dif.style.format({var: '{:,.2%}'.format}))
                    else:
                        if st.checkbox("Select Number of Items to be Display"):
                            n = st.number_input("Enter", 5)
                            rad = st.radio("Sort values by Order", ['Top', 'Bottom'])
                            if "Top" in rad:
                                st.dataframe(dif.head(n).sort_values(by=var, ascending=False))
                            else:
                                st.dataframe(dif.head(n).sort_values(by=var))
elif "Compare" in report:
    perf = st.sidebar.selectbox("Select column To Compare",
                                options=data.select_dtypes(exclude=['float', 'datetime64[ns]']).columns)
    val = st.sidebar.selectbox("Select filter to View", options=data[perf].value_counts().index)
    var = st.sidebar.selectbox("Select Quantitative field",
                               options=data.select_dtypes(include=['float'], exclude='datetime64[ns]').columns)
    metric = st.sidebar.selectbox("Select calculative field", options=['sum', 'max', 'min', 'mean', 'count'])
    cmpr = st.selectbox("Comparatitive Column", options=data.select_dtypes(exclude=['float', 'datetime64[ns]']).columns)
    grp = st.selectbox("Groupby Column",
                       options=sorted(data.select_dtypes(exclude=['float', 'datetime64[ns]']).columns, reverse=True))
    col1, col2 = st.columns(2)
    with col1:
        cmr = st.selectbox("Select value", options=np.sort(data[cmpr].value_counts().index))
        dtr = data.loc[(data[perf] == val) & (data[cmpr] == cmr)]
        res = dtr.groupby([grp]).agg({var: metric}).unstack(fill_value=0)
    with col2:
        cmr_1 = st.selectbox("Select value1", options=data[cmpr].value_counts().index)
        dtr_1 = data.loc[(data[perf] == val) & (data[cmpr] == cmr_1)]
        res1 = dtr_1.groupby([grp]).agg({var: metric}).unstack(fill_value=0)
    if 'Season' in grp:
        res1 = res1.droplevel(level=0)
        idx = ['Winter', 'Spring', 'Summer', 'Fall']
        res1.index = [idx[i - 1] for i in res1.index.values]
        res = res.droplevel(level=0)
        idx = ['Winter', 'Spring', 'Summer', 'Fall']
        res.index = [idx[i - 1] for i in res.index.values]
        st.table(pd.concat([res, res1], axis=1).rename(columns={0: cmr, 1: cmr_1}).fillna(0))
        st.table(pd.concat([res, res1], axis=1).rename(columns={0: cmr, 1: cmr_1}).fillna(0).sum())
    elif "Month" in grp:
        res1 = res1.droplevel(level=0)
        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        res1.index = [idx[i - 1] for i in res1.index.values]
        res = res.droplevel(level=0)
        idx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        res.index = [idx[i - 1] for i in res.index.values]
        st.table(pd.concat([res, res1], axis=1).rename(columns={0: cmr, 1: cmr_1}).fillna(0))
        st.table(pd.concat([res, res1], axis=1).rename(columns={0: cmr, 1: cmr_1}).fillna(0).sum())
    elif "Quarter" in grp:
        res1 = res1.droplevel(level=0)
        idx = ['Q1', 'Q2', 'Q3', 'Q4']
        res1.index = [idx[i - 1] for i in res1.index.values]
        res = res.droplevel(level=0)
        idx = ['Q1', 'Q2', 'Q3', 'Q4']
        res.index = [idx[i - 1] for i in res.index.values]
        st.table(pd.concat([res, res1], axis=1).rename(columns={0: cmr, 1: cmr_1}).fillna(0))
        st.table(pd.concat([res, res1], axis=1).rename(columns={0: cmr, 1: cmr_1}).fillna(0).sum())
    else:
        res2 = pd.concat([res, res1], axis=1).rename(columns={0: cmr, 1: cmr_1}).unstack(0).fillna(0)
        st.table(res2)
        st.table(pd.concat([res, res1], axis=1).rename(columns={0: cmr, 1: cmr_1}).fillna(0).sum())
elif "Multi-dimensional" in report:
    filter = st.sidebar.multiselect("Select a field",
                                    options=data.select_dtypes(exclude=['float64', 'datetime64[ns]']).columns,
                                    default='Customer')
    metric = st.sidebar.selectbox("Select a field",
                                  options=sorted(data.select_dtypes(exclude=['datetime64[ns]', 'int64']).columns,
                                                 reverse=True))
    st.table(mul_grp(data, filter, metric))
elif "Top Most" in report:
    filter = st.sidebar.selectbox("Select a field",
                                  options=data.select_dtypes(exclude=['datetime64[ns]', 'float64']).columns)
    value = st.sidebar.selectbox("Select a value", options=data[filter].value_counts().index)
    group = st.sidebar.selectbox("Select a groupby field",
                                 options=reversed(data.select_dtypes(exclude=['datetime64[ns]', 'float64']).columns))
    metric = st.sidebar.selectbox("Select a metric field", options=data.select_dtypes(include='object').columns)
    dwnld = top_most(data, filter, value, group, metric)
    st.write(dwnld)
    st.download_button("Download File", data=convert(dwnld), file_name="report.csv")
elif "Forecast" in report:
    itm = st.sidebar.selectbox("Select an Item Value", options=data['Item'].value_counts().index)
    metric = st.sidebar.selectbox("Select a Calculative Field", options=data.select_dtypes(include='float64').columns)
    ts = data.loc[data['Item'] == itm].groupby(['Date']).agg({metric: 'sum'}).resample('1D').sum()
    prd = st.number_input("Please Enter How many Days to Forecast", min_value=10)
    st.write("Forecasting...")
    st.table(time_series(ts, prd))
    st.line_chart(time_series(ts, prd))
elif "RFM" in report:
    st.subheader(
        "RFM analysis is a marketing technique used to quantitatively rank and group customers based on the recency, frequency and monetary total of their recent transactions to identify the best customers and perform targeted marketing campaigns.")
    tm = pd.to_datetime(st.sidebar.date_input("Select Date"))
    field = st.sidebar.selectbox("Select a Field",
                                 options=data.select_dtypes(exclude=['float64', 'datetime64[ns]', 'int64']).columns)
    variable = st.sidebar.selectbox("Select a Filter", options=data.select_dtypes(exclude=['float64', 'datetime64[ns]', 'int64']).columns)
    val = st.sidebar.selectbox('Select a value', options=data[variable].value_counts().index)
    result = rfm(data, field, variable, val, tm)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("High Valued" + " " + field)
        s = result.loc[result['Rating'] == 'High', result.columns[0]]
        ans = pd.DataFrame(s).style.hide_index()
        st.write(ans.to_html(), unsafe_allow_html=True)
    with col2:
        st.markdown("Medium valued" + " " + field)
        med = pd.DataFrame(result.loc[result['Rating'] == 'Medium', result.columns[0]]).style.hide_index()
        st.write(med.to_html(), unsafe_allow_html=True)
    with col3:
        st.markdown("Low valued" + " " + field)
        lw = pd.DataFrame(result.loc[result['Rating'] == 'Low', result.columns[0]]).style.hide_index()
        st.write(lw.to_html(), unsafe_allow_html=True)

