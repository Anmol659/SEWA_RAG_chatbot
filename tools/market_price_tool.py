import requests
from bs4 import BeautifulSoup
import sys # Import sys to get stdout encoding information

def get_mandi_prices(state: str, commodity: str) -> str:
    """
    Fetches and displays the latest mandi (market) prices for a specific
    agricultural commodity in a given state by scraping napanta.com. This tool
    is vital for helping farmers decide the best time and place to sell their produce.
    The model will extract the state and commodity from the user's query.

    Args:
        state: The state name, e.g., 'punjab'.
        commodity: The commodity name, e.g., 'wheat'.

    Returns:
        A formatted, human-readable string of the market price table, or an error message.
    """
    try:
        # Sanitize inputs for the URL
        state_formatted = state.strip().lower().replace(" ", "-")
        commodity_formatted = commodity.strip().lower().replace(" ", "-")
        url = f"https://www.napanta.com/agri-commodity-prices/{state_formatted}/{commodity_formatted}/"
        
        # Use a standard User-Agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise an error for bad status codes (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table")

        if not table:
            return f"Could not find a price table for '{commodity}' in '{state}'. Please check the spellings or try a different commodity."

        all_rows_data = []
        # Extract headers
        headers = [th.text.strip() for th in table.find_all("th")]
        if headers:
            all_rows_data.append(headers)

        # Extract data rows
        for tr in table.find_all("tr"):
            cols = [td.text.strip() for td in tr.find_all("td")]
            if cols:
                all_rows_data.append(cols)

        if len(all_rows_data) <= 1: # Only headers found or table is empty
            return f"No price data found for '{commodity}' in '{state}' at the moment."

        # --- Improved Table Formatting ---
        # Calculate the maximum width needed for each column for clean alignment
        num_columns = len(all_rows_data[0])
        col_widths = [0] * num_columns
        for row in all_rows_data:
            # Ensure row has the expected number of columns to prevent index errors
            if len(row) == num_columns:
                for i, cell in enumerate(row):
                    if len(cell) > col_widths[i]:
                        col_widths[i] = len(cell)
        
        # Build the formatted table string
        output = f"Mandi Prices for {commodity.title()} in {state.title()}:\n"
        # Header
        output += " | ".join(word.ljust(col_widths[i]) for i, word in enumerate(all_rows_data[0])) + "\n"
        # Separator line
        output += "-+-".join("-" * width for width in col_widths) + "\n"
        # Data Rows
        for row in all_rows_data[1:]:
             if len(row) == num_columns:
                output += " | ".join(word.ljust(col_widths[i]) for i, word in enumerate(row)) + "\n"
            
        return output.strip()

    except requests.exceptions.HTTPError:
        return f"Error: Could not fetch data. The page for '{commodity}' in '{state}' may not exist."
    except Exception as e:
        return f"An unexpected error occurred while fetching market prices: {e}"

if __name__ == "__main__":
    print("--- Testing Mandi Price Tool ---")
    state_input = "punjab"
    commodity_input = "wheat"
    price_report = get_mandi_prices(state_input, commodity_input)
    
    # FIX: This try-except block handles the printing issue on terminals with 
    # limited character support (like the default Windows command prompt).
    try:
        print(price_report)
    except UnicodeEncodeError:
        print("\nNote: Your terminal doesn't fully support Unicode characters like the Rupee symbol (₹).")
        print("Displaying with replacement characters:")
        # Encode to a universal format (UTF-8) and then decode to the terminal's
        # native format, replacing any characters it can't handle.
        print(price_report.encode('utf-8').decode(sys.stdout.encoding, 'replace'))
