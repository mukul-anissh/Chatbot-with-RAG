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

bonds = pd.read_csv("bonds.csv")
bonds.fillna("Unknown", inplace=True)

cashflows = pd.read_csv("cashflows.csv")
cashflows.fillna("Unknown", inplace=True)

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
        try:
            if not isinstance(coupon_details, str) or coupon_details == "Unknown":
                return None
            match = re.search(r'"couponRate":\s*"([\d.]+)%"', coupon_details)
            if match:
                return float(match.group(1))

        except Exception as e:
            print(f"Error extracting coupon rate: {e}")

        return None

    required_bonds["coupon_rate"] = required_bonds["coupon_details"].apply(extract_coupon_rate)

    for key, value in kwargs.items():
        if value:
            if key in ["isin", "issue_size"]:
                required_bonds = required_bonds[required_bonds[key] == value]
            elif key in ["credit_rating_details", "company_name"]:
                required_bonds = required_bonds[
                    required_bonds[key].astype(str).str.lower().str.contains(value.lower(), na=False)
                ]
            elif key == "allotment_condition":
                match = re.match(r"(before|after) (\d{4})", value)
                if match:
                    condition, year = match.groups()
                    year = int(year)
                    required_bonds = required_bonds[required_bonds["allotment_year"].astype(int) < year] if condition == "before" else required_bonds[required_bonds["allotment_year"].astype(int) > year]
            elif key == "maturity_condition":
                match = re.match(r"(before|after) (\d{4})", value)
                if match:
                    condition, year = match.groups()
                    year = int(year)
                    required_bonds = required_bonds[required_bonds["maturity_year"].astype(int) < year] if condition == "before" else required_bonds[required_bonds["maturity_year"].astype(int) > year]
            elif key == "coupon_condition":
                match = re.match(r"(above|below) (\d+(\.\d+)?)%", value)
                if match:
                    condition, rate = match.groups()[0], float(match.groups()[1])
                    required_bonds = required_bonds[required_bonds["coupon_rate"].astype(float) < rate] if condition == "below" else required_bonds[required_bonds["coupon_rate"].astype(float) > rate]
    
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

#setting up all agents
bond_agents = initialize_agent(
    tools=[bond_directory_agent_tool, bond_cashflow_agent_tool, bond_yield_calculator_tool],
    llm=model,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

#orchestrator agent
chat_history = [
    SystemMessage(
        "you are a financial AI assistant deployed by TapBonds, specializing in bonds and cashflows. "
        "use available tools to provide bond details, payment schedules, maturity dates, and investment insights. "
        "if a query is unrelated to bonds, politely inform the user. do not engage in those queries"
        "when a query is asked to you, imagine that query was not asked by the user and was asked by the admin. the admin wants the user to know the answer to it. so you must reply to the user as if he does not know the data he provided"
    ),
    HumanMessage("Hello")
]

greeting_response = model.invoke(chat_history)
chat_history.append(AIMessage(greeting_response.content))
print("AI:", greeting_response.content)

while True:
    query = input("You: ")
    if query.lower() == "q":
        break

    retrieve = bond_agents.invoke(query)

    if isinstance(retrieve, dict) and "output" in retrieve:
        agent_output = retrieve["output"]
    else:
        agent_output = retrieve

    formatted_query = f"Here is relevant data: {agent_output}. Now answer: {query}"
    chat_history.append(HumanMessage(formatted_query))

    response = model.invoke(chat_history)
    chat_history.append(AIMessage(response.content))
    print("\nAI:", response.content)
