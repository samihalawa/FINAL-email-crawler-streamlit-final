import os
import interpreter

def analyze_results():
    # Configure interpreter
    interpreter.model = "gpt-4o-mini"
    
    # Check if screenshot exists
    if not os.path.exists("search_results.png"):
        print("No search results screenshot found!")
        return
        
    # Create analysis prompt
    analysis_prompt = """
    I have a screenshot of search results from a web application at search_results.png.
    Please:
    1. Use computer vision to analyze the screenshot
    2. Tell me how many results were found
    3. List any interesting patterns in the results
    4. Suggest improvements for the search
    """
    
    # Get analysis from interpreter
    response = interpreter.chat(analysis_prompt)
    print("\nAnalysis complete!")
    
if __name__ == "__main__":
    print("Starting analysis of search results...")
    analyze_results() 