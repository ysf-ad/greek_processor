from datetime import datetime, date, timedelta
import requests
import json

class MarketDataService:
    """Handles all market data related operations"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:25510"  # Theta Terminal local server

    def get_current_spot_price(self, root):
        """Get current spot price for a symbol using Greeks snapshot"""
        try:
            # Get today's date in YYYYMMDD format
            today = date.today().strftime("%Y%m%d")
            
            # Build URL with required parameters
            url = f"{self.base_url}/v2/bulk_snapshot/option/greeks"
            params = {
                'root': root,
                'exp': today,  # Current expiry, will get under_price regardless
                'use_csv': 'false'  # Use JSON format
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and data['response']:
                    # Get under_price from the response
                    under_price = data['response'].get('under_price')
                    if under_price:
                        return float(under_price)
            
            print(f"Error getting spot price: status {response.status_code}")
            print(f"Response: {response.text}")
            return None
        except Exception as e:
            print(f"Error getting spot price: {str(e)}")
            return None

    def list_option_contracts(self, root):
        """Get list of available option contracts for a root"""
        try:
            # Get today's date in YYYYMMDD format
            today = date.today().strftime("%Y%m%d")
            
            # Build URL with required parameters
            url = f"{self.base_url}/v2/list/contracts/option/quote"
            params = {
                'root': root,
                'start_date': today
            }
            
            print(f"Getting contracts: {url} with params {params}")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error listing contracts: status {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error listing contracts: {str(e)}")
        return None

    def get_quote(self, contract):
        """Get latest quote for a specific contract"""
        try:
            # Format contract parameters
            root = contract['root']
            expiry = contract['expiration']
            strike = float(contract['strike'])
            right = contract['right']
            
            # Format strike properly (handle decimals)
            if strike < 1000:
                strike = int(strike * 1000)
            else:
                strike = int(strike)
            
            url = f"{self.base_url}/v2/hist/quote/option/{root}/{expiry}/{strike}/{right}/latest"
            print(f"Getting quote: {url}")  # Debug
            
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting quote: status {response.status_code}")
        except Exception as e:
            print(f"Error getting quote: {str(e)}")
        return None
        
