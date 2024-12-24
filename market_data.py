from datetime import datetime, date, timedelta
import requests
import json

class MarketDataService:
    """Handles all market data related operations"""
    
    @staticmethod
    def get_previous_trading_day():
        today = date.today()
        return today.strftime("%Y%m%d")

    def get_current_spot_price(self, root, target_date=None):
        """Get spot price for a symbol, defaulting to today"""
        today = date.today().strftime("%Y%m%d")
        return 590.0  # Default live price for now

    def get_historical_spot_price(self, root, target_date):
        """Get historical spot price - placeholder for now"""
        print(f"Getting historical price for {root} on {target_date}")
        return 590.0  # Default historical price for now

    @staticmethod
    def get_bulk_quote_data(symbol: str):
        """Fetch bulk quote data for a symbol"""
        today = date.today().strftime("%Y%m%d")
        url = f"http://127.0.0.1:25510/v2/bulk_hist/option/quote"
        
        params = {
            'root': symbol,
            'exp': 0,  # Get all expiries
            'start_date': today,
            'end_date': today,
            'ivl': 3600000  # 1-hour intervals
        }
        
        print(f"\nFetching bulk quotes for {symbol} on {today}")
        print(f"URL: {url}")
        print(f"Params: {params}")
        
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"Error: Got status code {response.status_code}")
                return {'response': []}
                
            data = response.json()
            if 'response' in data:
                contracts = data['response']
                print(f"Received {len(contracts)} contracts for {symbol}")
                
                # Convert bulk quote data to contract format
                formatted_contracts = []
                for contract_data in contracts:
                    # Get the latest quote (last tick in the array)
                    ticks = contract_data.get('ticks', [])
                    latest_tick = ticks[-1] if ticks else None
                    contract_info = contract_data.get('contract', {})
                    
                    if latest_tick and contract_info:
                        # Format based on header: ["ms_of_day","bid_size","bid_exchange","bid",
                        # "bid_condition","ask_size","ask_exchange","ask","ask_condition","date"]
                        formatted_contracts.append({
                            'contract': {
                                'root': contract_info.get('root'),
                                'expiration': str(contract_info.get('expiration')),
                                'strike': str(contract_info.get('strike')),
                                'right': contract_info.get('right')
                            },
                            'quote': {
                                'bid': latest_tick[3],  # bid price
                                'ask': latest_tick[7],  # ask price
                                'bid_size': latest_tick[1],  # bid size
                                'ask_size': latest_tick[5],  # ask size
                                'bid_exchange': latest_tick[2],  # bid exchange
                                'ask_exchange': latest_tick[6],  # ask exchange
                            }
                        })
                
                print(f"Formatted {len(formatted_contracts)} valid contracts")
                return {'response': formatted_contracts}
            else:
                print(f"No 'response' field in data")
                return {'response': []}
                
        except Exception as e:
            print(f"Error fetching bulk quotes: {str(e)}")
            return {'response': []}
        
