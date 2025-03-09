import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI as gemini
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from os import getenv
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional
import re
import datetime

load_dotenv()

model = gemini(model="gemini-2.0-flash", api_key=getenv("API_KEY"))

bonds = pd.read_csv("bonds.csv").fillna("Unknown")
cashflows = pd.read_csv("cashflows.csv").fillna("Unknown")

#bond directory agent
class BondQuery(BaseModel):
    isin: Optional[str] = None
    credit_rating: Optional[str] = None
    company_name: Optional[str] = None
    issue_size: Optional[str] = None
    allotment_condition: Optional[str] = None
    maturity_condition: Optional[str] = None
    coupon_condition: Optional[str] = None

def bond_info(**kwargs):
    required_bonds = bonds.copy()
    required_bonds['allotment_date'] = pd.to_datetime(required_bonds['allotment_date'], errors='coerce')
    required_bonds['allotment_year'] = required_bonds['allotment_date'].dt.year
    required_bonds['maturity_date'] = pd.to_datetime(required_bonds['maturity_date'], errors='coerce')
    required_bonds['maturity_year'] = required_bonds['maturity_date'].dt.year

    def extract_coupon_rate(coupon_details):
        match = re.search(r'"couponRate":\s*"([\d.]+)%"', str(coupon_details))
        return float(match.group(1)) if match else None

    required_bonds["coupon_rate"] = required_bonds["coupon_details"].apply(extract_coupon_rate)

    for key, value in kwargs.items():
        if value:
            if key in ["isin", "issue_size"]:
                required_bonds = required_bonds[required_bonds[key] == value]
            elif key in ["credit_rating_details", "company_name"]:
                required_bonds = required_bonds[required_bonds[key].str.contains(value, case=False, na=False)]
            elif key in ["allotment_condition", "maturity_condition"]:
                match = re.match(r"(before|after) (\d{4})", value)
                if match:
                    condition, year = match.groups()
                    year = int(year)
                    column = "allotment_year" if "allotment" in key else "maturity_year"
                    required_bonds = required_bonds[required_bonds[column] < year] if condition == "before" else required_bonds[required_bonds[column] > year]
            elif key == "coupon_condition":
                match = re.match(r"(above|below) (\d+\.?\d*)%", value)
                if match:
                    condition, rate = match.groups()
                    required_bonds = required_bonds[required_bonds["coupon_rate"] > float(rate)] if condition == "above" else required_bonds[required_bonds["coupon_rate"] < float(rate)]

    return required_bonds.to_dict(orient="records") if not required_bonds.empty else "No bonds found."

bond_directory_agent_tool = StructuredTool(
    name="bond_info",
    func=bond_info,
    description="Retrieve bond details using ISIN, credit rating, company name, issue size, etc.",
    args_schema=BondQuery
)

#bond cashflow agent
class CashflowQuery(BaseModel):
    isin: Optional[str] = None
    year: Optional[int] = None
    state: Optional[str] = None

def bond_cashflow_info(isin: Optional[str] = None, year: Optional[int] = None, state: Optional[str] = None):
    required_cashflows = cashflows.copy()

    if isin:
        required_cashflows = required_cashflows[required_cashflows["isin"] == isin]
    
    if year:
        required_cashflows["cash_flow_date"] = pd.to_datetime(required_cashflows["cash_flow_date"], errors="coerce")
        required_cashflows = required_cashflows[required_cashflows["cash_flow_date"].dt.year == year]
    
    if state:
        required_cashflows = required_cashflows[
            required_cashflows["state"].astype(str).str.lower().str.contains(state.lower(), na=False)
        ]

    return required_cashflows.to_dict(orient="records") if not required_cashflows.empty else "No cashflow data found."

bond_cashflow_agent_tool = StructuredTool(
    name="bond_cashflow_info",
    func=bond_cashflow_info,
    description="Retrieve bond cashflow details using ISIN, year, or state.",
    args_schema=CashflowQuery
)

#bond yield calculator agent
class BondYieldQuery(BaseModel):
    isin: Optional[str] = None
    issuer_name: Optional[str] = None
    investment_date: str
    units: int
    yield_rate: Optional[float] = None
    price: Optional[float] = None 

def calculate_bond_price_or_yield(
    isin: Optional[str] = None,
    issuer_name: Optional[str] = None,
    investment_date: str = None,
    units: int = None,
    yield_rate: Optional[float] = None,
    price: Optional[float] = None,
):
    if not isin and not issuer_name:
        return "Please provide either ISIN or issuer name."
    
    selected_bond = bonds.copy()
    if isin:
        selected_bond = selected_bond[selected_bond["isin"] == isin]
    elif issuer_name:
        selected_bond = selected_bond[selected_bond["company_name"].str.contains(issuer_name, case=False, na=False)]
    
    if selected_bond.empty:
        return "No bond found."
    
    bond_isin = selected_bond.iloc[0]["isin"]
    bond_cashflows = cashflows[cashflows["isin"] == bond_isin]
    bond_cashflows["cash_flow_date"] = pd.to_datetime(bond_cashflows["cash_flow_date"], errors="coerce")
    bond_cashflows = bond_cashflows.sort_values("cash_flow_date")
    
    if bond_cashflows.empty:
        return "No cashflow data found for the selected bond."
    
    try:
        investment_date = datetime.strptime(investment_date, "%Y-%m-%d")
    except ValueError:
        return "Invalid investment date format. Use YYYY-MM-DD."
    
    future_cashflows = bond_cashflows[bond_cashflows["cash_flow_date"] >= investment_date]
    if future_cashflows.empty:
        return "No future cashflows available."
    
    cashflow_dates = future_cashflows["cash_flow_date"].to_list()
    coupon_payments = future_cashflows["interest_amount"].to_numpy()
    principal_payments = future_cashflows["principal_amount"].to_numpy()
    total_cashflows = coupon_payments + principal_payments
    
    if yield_rate is not None:
        discount_factors = [(1 + yield_rate / 100) ** ((date - investment_date).days / 365) for date in cashflow_dates]
        bond_price = sum(cf / df for cf, df in zip(total_cashflows, discount_factors))
        return {"bond_price": bond_price * units}
    
    if price is not None:
        def yield_function(y):
            return sum(cf / ((1 + y / 100) ** ((date - investment_date).days / 365)) for cf, date in zip(total_cashflows, cashflow_dates)) - price
        
        yield_rate = np.roots([yield_function(y) for y in np.linspace(0, 50, 1000)])
        yield_rate = yield_rate[np.isreal(yield_rate)].real.min()
        return {"yield_rate": yield_rate}
    
    return "Please provide either yield rate or price to compute the required value."

bond_yield_calculator_tool = StructuredTool(
    name="calculate_bond_price_or_yield",
    func=calculate_bond_price_or_yield,
    description="Calculate bond price from yield or yield from price. Provide ISIN or issuer name, investment date, units, and either yield rate or price.",
    args_schema=BondYieldQuery
)

#all agents
bond_agents = initialize_agent(
    tools=[bond_directory_agent_tool, bond_cashflow_agent_tool, bond_yield_calculator_tool],
    llm=model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

#ui
st.title("TapBonds AI Chatbot")
st.write("Mukul Anissh G")

chat_history = [
    SystemMessage(
        "you are a financial AI assistant deployed by TapBonds, specializing in bonds and cashflows. "
        "use available tools to provide bond details, payment schedules, maturity dates, and investment insights. "
        "if a query is unrelated to bonds, politely inform the user. do not engage in those queries"
        "when a query is asked to you, imagine that query was not asked by the user and was asked by the admin. the admin wants the user to know the answer to it. so you must reply to the user as if he does not know the data he provided"
        "politely decline any inappropriate request." 
        "for any investment related advice you provide based on the given data, suggest the user to go talk to an expert on tapbonds.com"
        "1.0. What is the Bond Screener on Tap Bonds?"
        "The Bond Screener is a tool provided by Tap Bonds that allows investors to evaluate bond-issuing companies by analyzing key trends, financial ratios, and the latest news, enabling informed investment decisions."
        "2.0. How can I access the Bond Screener?"
        "To access the Bond Screener, navigate to the 'Tools' section on the Tap Bonds website and select 'Bond Screener'. You may need to sign in or create an account to use this feature at full potential."
        "3.0. What information can I analyze using the Bond Screener?"
        "The Bond Screener allows you to analyze key trends, financial ratios, and the latest news related to bond-issuing companies, helping you make informed investment decisions."
        "4.0. Is there a cost associated with using the Bond Screener?"
        "The information available does not specify whether there is a cost associated with using the Bond Screener. For detailed information, please visit the Tap Bonds website or contact their support team."
        "5.0. Can I use the Bond Screener to compare different bonds?"
        "Yes, the Bond Screener is designed to help you evaluate and compare different bonds by providing detailed analyses of bond-issuing companies."
        "6.0. Do I need an account to use the Bond Screener?"
        "Accessing the Bond Screener may require you to sign in or create an account on the Tap Bonds platform."
        "7.0. How frequently is the information in the Bond Screener updated?"
        "The information available does not specify the update frequency of the Bond Screener data. For detailed information, please visit the Tap Bonds website or contact their support team."
        "8.0. Can I save my screening criteria for future use?"
        "The information available does not specify whether you can save your screening criteria. For detailed information, please visit the Tap Bonds website or contact their support team."
        "9.0. Is the Bond Screener suitable for beginners?"
        "The Bond Screener is designed to assist investors in evaluating bond-issuing companies. If you're new to bond investing, you may also consider using the 'Talk to Expert' feature on Tap Bonds for personalized guidance."
        "10.0. Where can I find more information or assistance regarding the Bond Screener?"
        "For more information or assistance, you can visit the 'Knowledge Centre' on the Tap Bonds website, which includes FAQs and other resources, or use the 'Talk to Expert' feature for personalized support."
        "11.0. How many companies are there in Bond Screener?"
        "Count from the list of companies in bond screener"
        "12.0. What is the rating of ABC company?"
        "Fetch from table"
        "13.0. ABC company is in which sector?"
        "Fetch from table"
        "14.0. ABC company is in which industry?"
        "Fetch from table"
        "15.0. Give me a summary about ABC company"
        "Generate AI summary"
        "16.0. What is the EPS of ABC company?"
        "Fetch from Key metrics"
        "17.0. What is the current ratio of ABC company?"
        "Fetch from Key metrics"
        "18.0. What is the Debt/Equity of ABC company?"
        "Fetch from Key metrics"
        "19.0. What is the Debt/EBITDA of ABC company?"
        "Fetch from Key metrics"
        "20.0. What is the interest coverage ratio of ABC company?"
        "Fetch from Key metrics"
        "21.0. What is the operating cashflow to total Debt of ABC company?"
        "Fetch from Key metrics"
        "22.0. Compare X ratio/metric of ABC company with DEF company."
        "Provide in text/tabular and inform which one has better metric?"
        "23.0. What is the {Specicific financial metric} of ABC company?"
        "Fetch from Financials"
        "24.0. What is the growth rate of {Specific Financial Metric} of ABC company."
        "nan"
        "25.0. What are the PROS and CONS of ABC company?"
        "Fetch from Pros and cons section of a company page."
        "26.0. List of lenders of ABC limited."
        "Fetch from lenders section of a company page."
        "27.0. Top lenders of ABC limited"
        "Fetch from lenders section of a company page."
        "28.0. List of ISINs under ABC limited"
        "Fetch from isin section of a company page."
        "29.0. Recent news/events/article/blog about ABC company"
        "Fetch from news section of screener page, blogs from Knowledge centre."
        " "
        "Calculator"
        "30. How can I calculate clean price of a bond?"
        "ask them to use calculator from tool or ask for parameters to fill and calculate results."
        "31. How can I calculate consideration of a bond?"
        "ask them to use calculator from tool or ask for parameters to fill and calculate results."
        "32. What will be the consideration for {ISIN} for {units} at {date}?"
        "ask them to use calculator from tool or ask for parameters to fill and calculate results."
        "33. Price to yield calculation"
        "Calculate from calculator tool and generate result."
        ),
    HumanMessage("Hello")
    ]

greeting = model.invoke(chat_history)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(greeting.content)]

for msg in st.session_state.chat_history:
    role = "User" if isinstance(msg, HumanMessage) else "AI"
    st.chat_message(role).write(msg.content)

query = st.chat_input("Ask about bonds...")
if query:
    query = query.strip()
    chat_history.append(HumanMessage(query))
    st.session_state.chat_history.append(HumanMessage(query))
    st.chat_message("User").write(query)

    if query is None or query.strip() == "":
        print("skipping blank entries")
    else:
        try:
            retrieve = bond_agents.invoke(query)
        except:
            st.chat_message("AI").write("I'm having some trouble dealing with that query. Is there anything else I can help you with?")

    if bond_agents is None:
        st.chat_message("AI").write("agents is not there only da")
    agent_output = retrieve.get("output", retrieve) if isinstance(retrieve, dict) else retrieve
    formatted_query = f"Here is relevant data: {agent_output}. Now answer: {query}"
    chat_history.append(HumanMessage(formatted_query))
    response = model.invoke(chat_history)
    st.session_state.chat_history.append(AIMessage(response.content))
    chat_history.append(AIMessage(response.content))
    st.chat_message("AI").write(response.content)
