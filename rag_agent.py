"""
RAG Agent for Snowflake
Retrieval-Augmented Generation agent that can query Snowflake databases
"""

from snowflake_connector import SnowflakeRAGConnector
from typing import List, Dict, Any
import pandas as pd


class SnowflakeRAGAgent:
    """
    RAG Agent that uses Snowflake as a knowledge base
    """

    def __init__(self, connector: SnowflakeRAGConnector):
        """
        Initialize the RAG agent

        Args:
            connector: SnowflakeRAGConnector instance
        """
        self.connector = connector
        self.table_schemas = {}
        self.available_tables = []

    def initialize(self):
        """Initialize the agent by discovering available tables"""
        print("Initializing RAG Agent...")

        # Get list of available tables
        query = """
        SELECT TABLE_NAME, TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
        ORDER BY TABLE_NAME
        """

        try:
            tables_df = self.connector.execute_query(query)
            self.available_tables = tables_df['TABLE_NAME'].tolist()
            print(f"✓ Discovered {len(self.available_tables)} tables")

            # Cache table schemas for faster retrieval
            for table in self.available_tables:
                self.table_schemas[table] = self.connector.get_table_schema(table)

            return True

        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            return False

    def get_context_for_question(self, question: str) -> Dict[str, Any]:
        """
        Get relevant context from Snowflake for a given question

        Args:
            question: User question

        Returns:
            Dictionary with context information
        """
        # This is a simple implementation
        # In a full RAG system, you would use embeddings and semantic search

        context = {
            'available_tables': self.available_tables,
            'schemas': self.table_schemas,
            'question': question
        }

        return context

    def query_with_natural_language(self, question: str) -> str:
        """
        Process a natural language question and query Snowflake

        Args:
            question: Natural language question

        Returns:
            Response string with the answer
        """
        print(f"\nQuestion: {question}")
        print("-" * 50)

        # Get context
        context = self.get_context_for_question(question)

        # TODO: Implement LLM integration to convert question to SQL
        # For now, this is a placeholder that shows available context

        response = f"""
Available Tables: {', '.join(context['available_tables'])}

To implement full RAG functionality:
1. Add an LLM (OpenAI, Anthropic, etc.) to convert questions to SQL
2. Use embeddings to find relevant tables/columns
3. Execute generated SQL queries
4. Format results into natural language responses

Example workflow:
- Question → LLM → SQL Query → Snowflake → Results → LLM → Natural Language Answer
"""

        return response

    def execute_sql_with_explanation(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL and provide explanation

        Args:
            sql: SQL query string

        Returns:
            Dictionary with results and explanation
        """
        try:
            results = self.connector.execute_query(sql)

            return {
                'success': True,
                'query': sql,
                'row_count': len(results),
                'results': results,
                'columns': results.columns.tolist()
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'query': sql
            }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Snowflake RAG Agent Demo")
    print("=" * 60)

    # Initialize connector
    connector = SnowflakeRAGConnector()

    if connector.connect():
        try:
            # Create and initialize RAG agent
            agent = SnowflakeRAGAgent(connector)

            if agent.initialize():
                # Example interaction
                print("\n" + "=" * 60)
                question = "What tables are available in the database?"
                response = agent.query_with_natural_language(question)
                print(response)

                # Example SQL execution
                print("\n" + "=" * 60)
                print("Example: Direct SQL Execution")
                print("-" * 60)
                result = agent.execute_sql_with_explanation(
                    "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()"
                )

                if result['success']:
                    print(f"✓ Query executed successfully")
                    print(f"  Rows returned: {result['row_count']}")
                    print(f"\nResults:")
                    print(result['results'])
                else:
                    print(f"✗ Query failed: {result['error']}")

        finally:
            connector.close()
    else:
        print("\n✗ Could not connect to Snowflake")
        print("Please configure your .env file with valid credentials")
