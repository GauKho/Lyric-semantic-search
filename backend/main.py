from flask import Flask, render_template, request
import os
import traceback

# Add error handling for imports
try:
    from retrieval.hybrid_lyrics_searching import HybridLyricsSearch
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the 'retrieval' folder exists with hybrid_lyrics_searching.py")
    exit(1)

app = Flask(__name__)

# Better path handling
data_path = os.path.join("backend", "data", "csv")

# Check if data path exists
if not os.path.exists(data_path):
    print(f"Data path does not exist: {data_path}")
    print("Please check your data directory structure")
    exit(1)

# Initialize search engine with error handling
try:
    search_engine = HybridLyricsSearch(data_path=data_path)
    print("Search engine initialized successfully")
except Exception as e:
    print(f"Error initializing search engine: {e}")
    traceback.print_exc()
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""
    error_message = ""
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        
        if query:
            try:
                print(f"Searching for: {query}")
                results = search_engine.search(query, top_k=5)
                print(f"Found {len(results)} results")
            except Exception as e:
                error_message = f"Search failed: {str(e)}"
                print(f"Search error: {e}")
                traceback.print_exc()
        else:
            error_message = "Please enter a search query"
    
    return render_template('index.html', 
                         results=results, 
                         query=query, 
                         error_message=error_message)

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='127.0.0.1', port=5000)