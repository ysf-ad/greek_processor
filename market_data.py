import requests
import datetime
from typing import Dict, Optional
import json
import bisect

class MarketData:
    def __init__(self, root: str, date: str):
        self.root = root
        self.date = date
        self.spot_price_data: Dict[int, float] = {}  # ms_of_day -> price mapping

    def get_first_strike(self) -> Optional[int]:
        """Get the first available strike for the given root and date."""
        url = f"http://127.0.0.1:25510/v2/list/strikes?root={self.root}&exp={self.date}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error getting strikes: {response.status_code}")
            return None
            
        data = response.json()
        if not data["response"]:
            print("No strikes found")
            return None
            
        return data["response"][0]

    def load_spot_prices(self) -> None:
        """Load spot prices for the given root and date."""
        # First get a valid strike
        strike = self.get_first_strike()
        if not strike:
            print("Could not get valid strike")
            return
        
        # Construct URL for greeks endpoint
        url = f"http://127.0.0.1:25510/v2/hist/option/greeks?root={self.root}&exp={self.date}&strike={strike}&right=C&start_date={self.date}&end_date={self.date}&ivl=500"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error getting spot prices: {response.status_code}")
            return
            
        data = response.json()
        
        # Get indices from header format
        header_format = data["header"]["format"]
        MS_IDX = header_format.index("ms_of_day")
        PRICE_IDX = header_format.index("underlying_price")
        
        # Create mapping of ms_of_day to underlying_price
        self.spot_price_data = {
            tick[MS_IDX]: tick[PRICE_IDX]
            for tick in data["response"]
        }
        
        print(f"Loaded {len(self.spot_price_data)} spot price points")

    def get_spot_price(self, ms_of_day: int) -> Optional[float]:
        if ms_of_day not in self.spot_price_data:
            return None
        """Get spot price for the rounded ms_of_day."""
        rounded_ms = round(ms_of_day / 500) * 500
        return self.spot_price_data.get(rounded_ms)

    def get_day_trade_quotes(self) -> Optional[Dict]:
        """Retrieve trade and quote data for the day."""
        url = f"http://127.0.0.1:25510/v2/bulk_hist/option/trade_quote?root={self.root}&exp={self.date}&start_date={self.date}&end_date={self.date}&exclusive=true"
        
        try:
            print(f"Requesting initial data from: {url}")
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return None
            
            try:
                data = response.json()
                print(f"Initial data received with {len(data['response'])} responses")
            except requests.exceptions.JSONDecodeError as e:
                print(f"JSON decode error at position {e.pos}. Attempting to truncate and parse...")
                valid_json = response.text[:e.pos]
                last_bracket = valid_json.rfind(']')
                if last_bracket != -1:
                    valid_json = valid_json[:last_bracket+1] + ']}'
                    data = json.loads(valid_json)
                else:
                    print("Could not recover valid JSON")
                    return None
            
            all_responses = data["response"]
            print(f"got first page with {len(all_responses)} responses")
            
            while True:
                next_page = data["header"].get("next_page")
                if not next_page or next_page == "null":
                    print("No more pages to fetch.")
                    break
                
                print(f"Fetching next page: {next_page}")
                try:
                    response = requests.get(next_page)
                    if response.status_code != 200:
                        print(f"Error fetching next page: {response.status_code}")
                        break
                    
                    data = response.json()
                    all_responses.extend(data["response"])
                    print(f"got next page with {len(data['response'])} responses")
                    
                except requests.exceptions.JSONDecodeError as e:
                    print(f"Error on pagination, stopping here with {len(all_responses)} total responses")
                    break
                except Exception as e:
                    print(f"Unexpected error during pagination: {str(e)}")
                    break
            
            print(f"Total responses collected: {len(all_responses)}")
            return {"header": data["header"], "response": all_responses}
            
        except Exception as e:
            print(f"Error fetching trade quotes: {str(e)}")
            return None

    def get_day_trades(self) -> Optional[Dict]:
        """Get all trades for a given root on a specific day."""
        print(f"getting trades for {self.root} on {self.date}")
        url = f"http://127.0.0.1:25510/v2/bulk_hist/option/trade?root={self.root}&exp=0&start_date={self.date}&end_date={self.date}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return None
            
        data = response.json()
        all_responses = data["response"]
        print(f"got first page")
        
        # Handle pagination
        while True:
            next_page = data["header"].get("next_page")
            if next_page == "null" or next_page is None:
                break
                
            response = requests.get(next_page)
            if response.status_code != 200:
                print(f"Error fetching next page: {response.status_code}")
                break
            print(f"got next page")
                
            data = response.json()
            all_responses.extend(data["response"])
        
        return {"header": data["header"], "response": all_responses}


