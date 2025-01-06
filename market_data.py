import requests
import datetime
from typing import Dict, Optional
import json

class MarketData:
    def __init__(self):
        self.spot_price_data: Dict[int, float] = {}  # ms_of_day -> price mapping

    def get_first_strike(self, root: str, date: str) -> Optional[int]:
        """Get the first available strike for a given root and date"""
        url = f"http://127.0.0.1:25510/v2/list/strikes?root={root}&exp={date}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error getting strikes: {response.status_code}")
            return None
            
        data = response.json()
        if not data["response"]:
            print("No strikes found")
            return None
            
        return data["response"][0]

    def load_spot_prices(self, root: str, date: str) -> None:
        """Load spot prices for a given root and date"""
        # First get a valid strike
        strike = self.get_first_strike(root, date)
        if not strike:
            print("Could not get valid strike")
            return
        
        # Construct URL for greeks endpoint
        url = f"http://127.0.0.1:25510/v2/hist/option/greeks?root={root}&exp={date}&strike={strike}&right=C&start_date={date}&end_date={date}&ivl=500"
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
        """Get spot price closest to given ms_of_day"""
        if not self.spot_price_data:
            print("No spot price data loaded")
            return None
            
        # Find closest time
        closest_ms = min(self.spot_price_data.keys(), 
                        key=lambda x: abs(x - ms_of_day))
        
        return self.spot_price_data[closest_ms]

    @staticmethod
    def get_day_trade_quotes(root, day):
        # Initial request
        url = f"http://127.0.0.1:25510/v2/bulk_hist/option/trade_quote?root={root}&exp={day}&start_date={day}&end_date={day}&exclusive=true"
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return None
            
            # Try to parse JSON with error handling
            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError as e:
                print(f"JSON decode error at position {e.pos}. Attempting to truncate and parse...")
                # Try to parse the valid portion of the response
                valid_json = response.text[:e.pos]
                # Find the last complete record
                last_bracket = valid_json.rfind(']')
                if last_bracket != -1:
                    valid_json = valid_json[:last_bracket+1] + ']}'
                    data = json.loads(valid_json)
                else:
                    print("Could not recover valid JSON")
                    return None
            
            all_responses = data["response"]
            print(f"got first page with {len(all_responses)} responses")
            
            # Handle pagination
            while True:
                next_page = data["header"].get("next_page")
                if next_page == "null" or next_page is None:
                    break
                    
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
            
            return {"header": data["header"], "response": all_responses}
            
        except Exception as e:
            print(f"Error fetching trade quotes: {str(e)}")
            return None

    @staticmethod
    def get_day_trades(root: str, day: str):
        print(f"getting trades for {root} on {day}")
        """Get all trades for a given root on a specific day"""
        url = f"http://127.0.0.1:25510/v2/bulk_hist/option/trade?root={root}&exp=0&start_date={day}&end_date={day}"
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


