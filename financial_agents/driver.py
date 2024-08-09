from workflows.Q1_financial_analyser_agent import Q1FinancialAnalyser
from workflows.Q2_financial_analyser_agent import Q2FinancialAnalyser
from workflows.Q3_financial_analyser_agent import Q3FinancialAnalyser
from workflows.Q4_financial_analyser_agent import Q4FinancialAnalyser
from workflows.annual_financial_analyser_agent import AnnualFinancialAnalyser
import nest_asyncio

# Apply the nest_asyncio
nest_asyncio.apply()


async def main():
    w1 = Q1FinancialAnalyser(timeout=300, verbose=True)
    w2 = Q2FinancialAnalyser(timeout=300, verbose=True)
    w3 = Q3FinancialAnalyser(timeout=300, verbose=True)
    w4 = Q4FinancialAnalyser(timeout=300, verbose=True)
    final_summary_analyser = AnnualFinancialAnalyser(timeout=300, verbose=True)

    user_query = ("What was the Reconciliation of selected GAAP-based measures to Non-GAAP-based "
                  "measures for the nine months")

    q1_result = await w1.run(user_query=user_query)
    q2_result = await w2.run(user_query=user_query)
    q3_result = await w3.run(user_query=user_query)
    q4_result = await w4.run(user_query=user_query)

    final_summary = await final_summary_analyser.run(individual_summaries=[q1_result, q2_result, q3_result, q4_result])

    print(final_summary)

if __name__ == '__main__':
    import asyncio

    asyncio.run(main=main())
