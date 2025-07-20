from tuik_scraper import TuikScraper
import os
import json
import pandas as pd
import asyncio
import logging
import httpx
import click
import jwt
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import namedtuple
from mcp.server.fastmcp import FastMCP
from utils.logging import setup_logger
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair

# --- Configuration ---
PUBLIC_KEY_FILE = "public_key.pem"
ISSUER_URL = "https://wagmi.tech/auth"
AUDIENCE = "wagmi-tech-payment-link-mcp-server"

AuthInfo = namedtuple("AuthInfo", ["claims", "expires_at", "scopes", "client_id"])

class SimpleBearerAuthProvider:
    def __init__(self, public_key: bytes, issuer: str, audience: str):
        self.public_key = public_key
        self.issuer = issuer
        self.audience = audience
        self.logger = setup_logger(__name__)

    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify the token's signature, expiry, and claims."""
        try:
            decoded_token = jwt.decode(
                token,
                self.public_key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer,
            )
            client_id = decoded_token.get("sub")
            # The middleware expects an object with 'scopes', 'expires_at' and 'client_id' attributes.
            return AuthInfo(claims=decoded_token, expires_at=decoded_token.get("exp"), scopes=[], client_id=client_id)
        except jwt.PyJWTError as e:
            self.logger.error(f"Token verification failed: {e}")
            raise Exception("Invalid token")


class ConfigurationError(Exception):
    """Configuration error exception"""
    pass

class PaymentMCPServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8060, transport: str = "stdio", auth_token: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.mcp = None
        self.host = host
        self.port = port
        self.transport = transport
        self.auth_token = auth_token
    
    async def initialize(self) -> FastMCP:
        """Initialize the MCP server and provider."""
        try:
            self.logger.info(f"Initializing MCP server")

            # Setup authentication
            auth_provider = None
            if self.transport == 'sse':
                self.logger.info("SSE transport: setting up Simple Bearer authentication.")
                try:
                    with open(PUBLIC_KEY_FILE, "rb") as f:
                        public_key = f.read()
                    
                    auth_provider = SimpleBearerAuthProvider(
                        public_key=public_key,
                        issuer=ISSUER_URL,
                        audience=AUDIENCE
                    )
                    self.logger.info("Authentication provider loaded with public key.")
                except FileNotFoundError:
                    self.logger.error(f"{PUBLIC_KEY_FILE} not found. Please run dashboard.py to generate it.")
                    raise ConfigurationError(f"{PUBLIC_KEY_FILE} not found.")

            # Create MCP server
            auth_config = None
            if auth_provider:
                resource_server_url = f"http://{self.host}:{self.port}"
                auth_config = {
                    "issuer_url": ISSUER_URL,
                    "resource_server_url": resource_server_url,
                }

            self.mcp = FastMCP(
                name="TUIK Statistics MCP Server",
                host=self.host,
                port=self.port,
                token_verifier=auth_provider,
                auth=auth_config,
            )
            
            # Register tools
            self._register_tools()
            
            self.logger.info("MCP server initialized successfully")
            return self.mcp
            
        except Exception as e:
            self.logger.error(f"Failed to initialize server: {str(e)}")
            raise

    # Load available files data
    def load_data_from_json(self):
        """Load the available data from data.json"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, 'data.json')
            
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error("data.json not found. Please generate it first.")
            return []

    def _register_tools(self):
        """Register MCP tools."""
        @self.mcp.tool()
        async def analyze_question_and_select_files(user_question: str) -> str:
            """
            Analyze user's question about Turkish statistics and select relevant files.
            
            Args:
                user_question: The user's question about Turkish statistics
                
            Returns:
                JSON string containing selected file names and their category based on the question
            """
            try:
                # Load available files from data.json
                all_data = self.load_data_from_json()
                if not all_data:
                    return json.dumps({"error": "No data available from data.json"}, ensure_ascii=False, indent=2)

                all_files_with_category = []
                for category_data in all_data:
                    kategori_name = category_data.get("name")
                    kategori_path = category_data.get("kategori")
                    for file_name in category_data.get("files", []):
                        all_files_with_category.append({
                            "file_name": file_name,
                            "category_name": kategori_name,
                            "category_path": kategori_path
                        })

                # Create a comprehensive prompt for LLM to analyze the question and select relevant files
                llm_prompt = f"""
                TASK: Analyze the user's question about Turkish statistics and select the most relevant files from a SINGLE category.
                
                USER QUESTION: {user_question}
                
                AVAILABLE FILES (Total: {len(all_files_with_category)}):
                {json.dumps(all_files_with_category, ensure_ascii=False, indent=2)}
                
                INSTRUCTIONS:
                1. Carefully analyze the user's question to understand what specific data they need.
                2. Select files from ONLY ONE category that best answers the question.
                3. Prioritize files that directly address the question over general statistics.
                
                RESPONSE FORMAT:
                Return ONLY a valid JSON object with this exact structure:
                {{
                    "selected_files": ["file1.xls", "file2.xls"],
                    "category_path": "path_to_category_folder",
                    "category_name": "Full Category Name",
                    "reasoning": "Detailed explanation of why these specific files from this category were selected.",
                    "confidence": "high|medium|low",
                    "coverage": "complete|partial|limited"
                }}
                
                IMPORTANT:
                - Select between 1-5 files maximum from a single category.
                - File names must match exactly from the available list.
                """
                
                # Placeholder for LLM response
                placeholder_category = all_data[0] if all_data else {}
                selected_files = placeholder_category.get("files", [])[:5]

                result = {
                    "user_question": user_question,
                    "llm_prompt": llm_prompt,
                    "selected_files": selected_files,
                    "category_path": placeholder_category.get("kategori"),
                    "category_name": placeholder_category.get("name"),
                    "reasoning": "This is a placeholder response. Replace with actual LLM selection logic.",
                    "confidence": "low",
                    "coverage": "placeholder"
                }
                
                return json.dumps(result, ensure_ascii=False, indent=2)
                
            except Exception as e:
                error_result = {
                    "error": f"Failed to analyze question: {str(e)}",
                    "selected_files": [],
                    "reasoning": "Error occurred during analysis"
                }
                return json.dumps(error_result, ensure_ascii=False, indent=2)

        @self.mcp.tool()
        async def read_and_convert_files(selected_files_json: str) -> str:
            """
            Reads selected local Excel files and converts data to JSON format.
            
            Args:
                selected_files_json: JSON string or dict containing the list of selected files and their category.
                
            Returns:
                JSON string containing the converted data and analysis.
            """
            try:
                # Get the script directory for file operations
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Parse the selected files - handle both dict and string inputs
                if isinstance(selected_files_json, dict):
                    selected_data = selected_files_json
                elif isinstance(selected_files_json, str):
                    selected_data = json.loads(selected_files_json)
                else:
                    return json.dumps({
                        "error": f"Invalid input type: {type(selected_files_json)}. Expected dict or JSON string.",
                        "data": {}
                    }, ensure_ascii=False, indent=2)
                
                selected_files = selected_data.get("selected_files", [])
                kategori_path = selected_data.get("category_path")
                
                if not selected_files or not kategori_path:
                    return json.dumps({
                        "error": "No files or category path selected for processing",
                        "data": {},
                        "input_data": selected_data
                    }, ensure_ascii=False, indent=2)
                
                processed_data = {}
                successful_reads = 0
                failed_reads = []
                
                for file_name in selected_files:
                    try:
                        file_path = os.path.join(script_dir, 'data', kategori_path, file_name)
                        
                        if os.path.exists(file_path):
                            # Convert Excel to JSON
                            if file_path.endswith(('.xlsx', '.xls')):
                                try:
                                    if file_path.endswith('.xlsx'):
                                        df = pd.read_excel(file_path, engine='openpyxl')
                                    else:
                                        df = pd.read_excel(file_path, engine='xlrd')
                                    
                                    json_data = df.to_dict('records')
                                    
                                    metadata = {
                                        "rows": len(df),
                                        "columns": len(df.columns),
                                        "column_names": df.columns.tolist(),
                                        "file_path": file_path,
                                        "file_size_bytes": os.path.getsize(file_path),
                                        "file_extension": os.path.splitext(file_path)[1]
                                    }
                                    
                                    processed_data[file_name] = {
                                        "data": json_data,
                                        "metadata": metadata
                                    }
                                    
                                    successful_reads += 1
                                    self.logger.info(f"Successfully processed: {file_name}")
                                    
                                except Exception as excel_error:
                                    failed_reads.append({
                                        "file_name": file_name,
                                        "error": f"Excel conversion failed: {str(excel_error)}",
                                        "file_path": file_path
                                    })
                            else:
                                failed_reads.append({
                                    "file_name": file_name,
                                    "error": f"File is not an Excel file: {file_path}",
                                    "file_path": file_path
                                })
                        else:
                            failed_reads.append({
                                "file_name": file_name,
                                "error": "File not found at specified path",
                                "checked_path": file_path
                            })
                            
                    except Exception as process_error:
                        failed_reads.append({
                            "file_name": file_name,
                            "error": f"File processing error: {str(process_error)}"
                        })
                
                summary = {
                    "processing_summary": {
                        "total_requested": len(selected_files),
                        "successful_reads": successful_reads,
                        "failed_reads": len(failed_reads),
                        "success_rate": f"{(successful_reads/len(selected_files)*100):.1f}%" if selected_files else "0%"
                    },
                    "data": processed_data,
                    "failed_files": failed_reads,
                    "category_path": kategori_path,
                }
                
                if successful_reads > 0:
                    data_summary = {}
                    for file_name, file_data in processed_data.items():
                        data_summary[file_name] = {
                            "sample_data": file_data["data"][:5],
                            "metadata": file_data["metadata"]
                        }
                    
                    summary["data_summary_for_llm"] = data_summary
                    summary["llm_instruction"] = """
                    The above data has been successfully processed from local files.
                    Please analyze this data and provide insights that answer the user's original question.
                    Focus on key trends, patterns, and relevant statistics.
                    """
                
                return json.dumps(summary, ensure_ascii=False, indent=2, default=str)
                
            except Exception as e:
                error_result = {
                    "error": f"Failed to read and convert files: {str(e)}",
                    "data": {},
                    "input_data": str(selected_files_json),
                }
                return json.dumps(error_result, ensure_ascii=False, indent=2)

@click.command()
@click.option('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
@click.option('--port', default=8070, help='Server port (default: 8060)')
@click.option('--transport', envvar='TRANSPORT', default='sse', help='Transport type (default: stdio)')
@click.option('--auth-token', envvar='AUTH_TOKEN', help='Bearer token for SSE transport.')
def main(host, port, transport, auth_token):
    """Start the TUIK Employment Statistics MCP server."""
    
    logger = setup_logger(__name__)
    
    try:
        valid_transports = ['stdio', 'sse']
        if transport not in valid_transports:
            raise ConfigurationError(
                f"Unsupported transport '{transport}'. Available: {', '.join(valid_transports)}"
            )
        
        logger.info(f"Starting TUIK Employment Statistics MCP server")
        logger.info(f"Transport: {transport}")
        logger.info(f"Server will run on {host}:{port}")
        
        async def _run():
            server = PaymentMCPServer(host=host, port=port, transport=transport, auth_token=auth_token)
            mcp = await server.initialize()
            logger.info("MCP server started successfully")
            return mcp
        
        # Run the server
        mcp = asyncio.run(_run())
        mcp.run(transport=transport)
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    main()

    
    
    