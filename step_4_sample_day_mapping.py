import pickle
import pandas as pd

# 1st function - finding to which day all historical messages of a sample belongs to from the daily_h_l.csv file. i give it the sample_id and it returns the day.

def find_sample_day(sample_id, m_dict, daily_h_l_path, indexes_match=500):
    """
    Find which day a sample belongs to from the daily_h_l.csv file.
    
    Parameters
    ----------
    sample_id : int
        The sample ID to look up
    m_dict : dict
        Dictionary containing sample data
    daily_h_l_path : str
        Path to the daily_h_l.csv file
    indexes_match : int
        Number of indices to match
        
    Returns
    -------
    tuple or None
        Tuple of (day, highest_price, lowest_price, execution_sum) or None if not found
    """
    try:
        # Load the daily high-low data
        daily_df = pd.read_csv(daily_h_l_path)

        sample_indices = [x[0] for x in m_dict[sample_id]][1:][:indexes_match]
        print(f'Sample {sample_id} - len(sample_indices): {len(sample_indices)}, sample_indices[:{indexes_match}]: {sample_indices[:indexes_match]}')
        
        # Check each day in the CSV
        for _, row in daily_df.iterrows():
            day = row.values[0]
            indices_values_str = row.values[3]
            # Parse the string as a list and convert each value to int
            indices_values = [int(x) for x in eval(indices_values_str)]
            print(f'{_} Day {day} - len(indices_values): {len(indices_values)}, indices_values[:{indexes_match*2}]: {indices_values[:indexes_match*2]}')
            
            # Check if all sample indices are present in this day's indices
            if all(idx in indices_values for idx in sample_indices):
                print(f"All sample indices found in day {day}")
                highest_price = row.values[1]
                lowest_price = row.values[2]
                execution_sum = row.values[4]
                print(f'{_} Day {day} - filename: {row.values[0]}, highest_price: {row.values[1]}, lowest_price: {row.values[2]}, execution_sum: {row.values[4]}')
                return day, highest_price, lowest_price, execution_sum
            else:
                # Count how many indices match for debugging
                matches = sum(1 for idx in sample_indices if idx in indices_values)
                print(f"{_} Day {day}: {matches}/{len(sample_indices)} indices match")
        
        # If sample not found in any day
        print(f"Sample {sample_id} not found in any day")
        return None
        
    except FileNotFoundError:
        print(f"File {daily_h_l_path} not found")
        return None
    except Exception as e:
        print(f"Error reading daily data: {e}")
        return None


def sample_day_mapping(m_dict, daily_h_l_path, indexes_match=500):
    """
    Create a mapping of samples to their corresponding days.
    
    Parameters
    ----------
    m_dict : dict
        Dictionary containing sample data
    daily_h_l_path : str
        Path to the daily_h_l.csv file
    indexes_match : int
        Number of indices to match
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sample mappings
    """
    records = []
    sample_ids = sorted(m_dict.keys())

    for sid in sample_ids:
        print("="*80)
        print(f"Processing sample_id={sid}...")

        try:
            day_result = find_sample_day(sid, m_dict, daily_h_l_path, indexes_match)
            if day_result is not None:
                day, highest_price, lowest_price, execution_sum = day_result
                print(f"✅ Match found for sample {sid}: {day}, H={highest_price}, L={lowest_price}, V={execution_sum}")
                
                records.append({
                    "sample_id": sid,
                    "file_name": day,
                    "highest_price": highest_price,
                    "lowest_price": lowest_price,
                    "execution_sum": execution_sum
                })
            else:
                print(f"⚠️ No match found for sample {sid}")
                records.append({
                    "sample_id": sid,
                    "file_name": None,
                    "highest_price": float('nan'),
                    "lowest_price": float('nan'),
                    "execution_sum": 0
                })
        except Exception as e:
            print(f"❌ Error matching sample {sid}: {e}")
            records.append({
                "sample_id": sid,
                "file_name": None,
                "highest_price": float('nan'),
                "lowest_price": float('nan'),
                "execution_sum": 0
            })

    sample_day_df = pd.DataFrame(records)
    print(sample_day_df.head(20))
    return sample_day_df
    

def main():
    with open('/app/data_saved/exp_131_20250923_104218_gen_buy_300_b0_b7/m_dict.pkl', 'rb') as f:
        m_dict = pickle.load(f)
    daily_h_l_path = "/app/daily_h_l.csv"

    sample_day_df = sample_day_mapping(m_dict, daily_h_l_path, indexes_match=6)
    sample_day_df.to_csv("batches_equal_sample_day_map.csv", index=False)

    print("="*80)
    print(f"✅ Saved mapping for {len(sample_day_df)} samples → sample_day_map.csv")
    print("="*80)
    

if __name__ == "__main__":
    main()
