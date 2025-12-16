# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc

from time_util.time_util import TimeUtil
import traceback
from typing import Union, List


class ErrorUtils:
    """Shared utilities for error handling and formatting across the codebase."""
    
    @staticmethod
    def get_compact_stacktrace(error: Union[str, Exception], 
                             relevant_keywords: List[str] = None,
                             max_lines: int = 10) -> str:
        """
        Returns a compact stack trace showing only the most relevant frames.
        
        Args:
            error: Either an exception object or a string traceback
            relevant_keywords: List of keywords to identify relevant code paths
            max_lines: Maximum number of lines to return
            
        Returns:
            A compact, formatted stacktrace string suitable for Slack or logs
        """
        # Default keywords if none provided
        if relevant_keywords is None:
            relevant_keywords = [
                'metagraph', 'miner', 'validator', 'vali_', 'neurons/',
                'position', 'elimination', 'challengeperiod', 'weight', 
                'mdd_checker', 'auto_sync', 'p2p_syncer', 'plagiarism'
            ]
        
        # Get traceback lines
        if isinstance(error, Exception):
            tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        else:
            tb_lines = error.strip().split('\n')
        
        # Filter out lines that are less relevant
        relevant_lines = []
        for line in tb_lines:
            line_stripped = line.strip()
            # Keep the exception line and lines from our code
            if (line_stripped.startswith('File') and
                    (any(keyword in line for keyword in relevant_keywords) or
                     line.count('/') <= 3)):  # Keep shorter paths (likely our code)
                relevant_lines.append(line_stripped)
            elif not line_stripped.startswith('File'):
                # Keep error messages and exception types
                relevant_lines.append(line_stripped)
        
        # If we filtered too much, just take the last few lines
        if len(relevant_lines) < 3:
            relevant_lines = [line.strip() for line in tb_lines[-5:]]
        
        # Limit to specified max lines to keep it concise
        return '\n'.join(relevant_lines[-max_lines:])
    
    @staticmethod
    def get_operation_from_traceback(traceback_str: str) -> str:
        """
        Attempts to identify which operation failed based on the traceback content.
        
        Args:
            traceback_str: The full traceback string
            
        Returns:
            A descriptive name of the operation that failed
        """
        operation_map = {
            "price_slippage_model": "Price slippage model refresh",
            "position_syncer": "Position sync",
            "auto_sync": "AutoSync operation",
            "mdd_checker": "MDD check",
            "challengeperiod_manager": "Challenge period refresh",
            "elimination_manager": "Elimination processing",
            "weight_setter": "Weight setting",
            "p2p_syncer": "P2P sync",
            "metagraph": "Metagraph update",
            "plagiarism": "Plagiarism detection",
            "perf_ledger": "Performance ledger update",
            "live_price_fetcher": "Live price fetching",
            "api_manager": "API operation"
        }
        
        for key, operation in operation_map.items():
            if key in traceback_str.lower():
                return operation
        
        return "Unknown operation"
    
    @staticmethod
    def format_error_for_slack(error: Exception, 
                             traceback_str: str = None,
                             include_operation: bool = True,
                             include_timestamp: bool = True) -> str:
        """
        Formats an error for Slack notification with all relevant details.
        
        Args:
            error: The exception object
            traceback_str: Optional full traceback string
            include_operation: Whether to try to identify the failing operation
            include_timestamp: Whether to include current timestamp
            
        Returns:
            A formatted error message suitable for Slack
        """

        if traceback_str is None:
            traceback_str = traceback.format_exc()
        
        message_parts = []
        
        if include_operation:
            operation = ErrorUtils.get_operation_from_traceback(traceback_str)
            message_parts.append(f"Operation: {operation}")
        
        message_parts.append(f"Error: {str(error)}")
        
        if include_timestamp:
            message_parts.append(f"Time: {TimeUtil.millis_to_formatted_date_str(TimeUtil.now_in_millis())}")
        
        # Add compact stacktrace
        compact_trace = ErrorUtils.get_compact_stacktrace(traceback_str)
        message_parts.append(f"```{compact_trace}```")
        
        return "\n".join(message_parts)