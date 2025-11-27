"""
Snowflake Connector for RAG Agent
Handles authentication and connection to Snowflake
"""

import os
import snowflake.connector
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import pandas as pd


class SnowflakeRAGConnector:
    """
    A connector class for Snowflake that supports multiple authentication methods
    """

    def __init__(self, config: Optional[Dict[str, str]] = None):
        """
        Initialize Snowflake connector

        Args:
            config: Optional dictionary with connection parameters.
                   If not provided, will load from environment variables.
        """
        # Load environment variables
        load_dotenv()

        self.config = config or self._load_config_from_env()
        self.connection = None
        self.cursor = None

    def _load_config_from_env(self) -> Dict[str, str]:
        """Load configuration from environment variables"""
        return {
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'user': os.getenv('SNOWFLAKE_USER'),
            'password': os.getenv('SNOWFLAKE_PASSWORD'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
            'database': os.getenv('SNOWFLAKE_DATABASE'),
            'schema': os.getenv('SNOWFLAKE_SCHEMA'),
            'role': os.getenv('SNOWFLAKE_ROLE'),
        }

    def connect(self) -> bool:
        """
        Establish connection to Snowflake

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Basic authentication
            self.connection = snowflake.connector.connect(
                account=self.config['account'],
                user=self.config['user'],
                password=self.config['password'],
                warehouse=self.config.get('warehouse'),
                database=self.config.get('database'),
                schema=self.config.get('schema'),
                role=self.config.get('role'),
            )

            self.cursor = self.connection.cursor()
            print(f"✓ Successfully connected to Snowflake account: {self.config['account']}")
            return True

        except Exception as e:
            print(f"✗ Failed to connect to Snowflake: {str(e)}")
            return False

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame

        Args:
            query: SQL query string

        Returns:
            pandas DataFrame with query results
        """
        if not self.connection:
            raise ConnectionError("Not connected to Snowflake. Call connect() first.")

        try:
            self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()
            return pd.DataFrame(data, columns=columns)

        except Exception as e:
            print(f"✗ Query execution failed: {str(e)}")
            raise

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a table

        Args:
            table_name: Name of the table

        Returns:
            List of dictionaries containing column information
        """
        query = f"""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """

        df = self.execute_query(query)
        return df.to_dict('records')

    def get_sample_data(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """
        Get sample data from a table

        Args:
            table_name: Name of the table
            limit: Number of rows to fetch (default: 10)

        Returns:
            pandas DataFrame with sample data
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)

    def close(self):
        """Close the Snowflake connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("✓ Snowflake connection closed")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Example usage
if __name__ == "__main__":
    # Test the connector
    print("Testing Snowflake Connection...")
    print("-" * 50)

    connector = SnowflakeRAGConnector()

    if connector.connect():
        try:
            # Test query
            result = connector.execute_query("SELECT CURRENT_VERSION(), CURRENT_USER(), CURRENT_ROLE()")
            print("\nConnection Details:")
            print(result)

        except Exception as e:
            print(f"Error during testing: {e}")

        finally:
            connector.close()
    else:
        print("\nPlease check your .env file and ensure all credentials are correct.")
        print("See .env.example for the required format.")
