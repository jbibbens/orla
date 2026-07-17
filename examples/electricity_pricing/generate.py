import gridstatus
import pandas as pd

def collect_hub_lmp_data(target_date):
    """
    Collects 5-minute real-time LMP data for major hubs across CAISO, MISO, ISONE, and PJM.
    """
    
    # Initialize the specific ISOs 
    iso_mapping = {
        "CAISO": gridstatus.CAISO(),
        "MISO": gridstatus.MISO(),
        "ISONE": gridstatus.ISONE(),
        # "PJM": gridstatus.PJM()
    }
    
    # We use the standardized 5-minute market constant
    market = "REAL_TIME_5_MIN"
    all_hub_data = []

    for name, iso_obj in iso_mapping.items():
        print(f"Fetching {name} data for {target_date}...")
        try:
            if name == "CAISO":
                # For CAISO, passing no locations automatically defaults to the 
                # 3 major trading hubs: TH_NP15, TH_SP15, and TH_ZP26
                df = iso_obj.get_lmp(date=target_date, market=market)
            else:
                # For others, we pull all locations and then filter down to Hubs
                df = iso_obj.get_lmp(date=target_date, market=market, locations="ALL")
                
                # Filter to only keep rows where the Location Type is a "Hub"
                if 'Location Type' in df.columns:
                    df = df[df['Location Type'].str.contains('Hub', case=False, na=False)]
            
            # Identify the grid location
            df['Grid'] = name
            
            # Standardize timezones to UTC to allow for clean concatenation and comparison
            for time_col in ['Time', 'Interval Start', 'Interval End']:
                if time_col in df.columns:
                    df[time_col] = pd.to_datetime(df[time_col], utc=True)
                    
            all_hub_data.append(df)
            print(f" -> Successfully retrieved {len(df)} rows for {name} hubs.")
            
        except Exception as e:
            print(f" -> Failed to fetch {name}: {e}")
            
    if all_hub_data:
        # Combine all ISO dataframes into one master dataframe
        final_df = pd.concat(all_hub_data, ignore_index=True)
        return final_df
    return pd.DataFrame()

if __name__ == "__main__":
    # pass "today", "latest", or a specific date string like "2024-03-15"
    date_to_fetch = "today" 
    
    print(f"Starting data collection...\n")
    lmp_df = collect_hub_lmp_data(date_to_fetch)
    
    if not lmp_df.empty:
        print("\nData Collection Complete! First 5 rows:")
        print(lmp_df[['Time', 'Grid', 'Location', 'LMP']].head())
        
        # Save 
        output_filename = "5min_hub_lmps.csv"
        lmp_df.to_csv(output_filename, index=False)
        print(f"\nData successfully saved to {output_filename}")
    else:
        print("\nNo data was collected. Please check your dates and connection.")