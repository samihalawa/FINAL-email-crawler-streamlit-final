from search_strategy_tool import CustomerProfileSearchTool
import json

def analyze_high_value_segments():
    tool = CustomerProfileSearchTool()
    
    # Priority segments for 15-hour AI training package
    high_value_combinations = [
        ("business_owner", "productivity"),
        ("business_owner", "data_analysis"),
        ("manager", "data_analysis"),
        ("consultant", "productivity"),
        ("consultant", "marketing")
    ]
    
    results = {}
    for profile, pain_point in high_value_combinations:
        search_terms = tool.generate_combined_search_terms(profile, pain_point)
        results[f"{profile}_{pain_point}"] = search_terms
    
    return results

if __name__ == "__main__":
    results = analyze_high_value_segments()
    print("High-Value Customer Search Terms:")
    print(json.dumps(results, indent=2, ensure_ascii=False)) 