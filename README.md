# Wikipedia Analysis Tools

A collection of tools for analyzing Wikipedia edit history, detecting bias patterns, and tracking content changes across revisions.

## Tools

### 1. Wiki Blame (`wiki_blame.py`)
Git-blame-style tool for Wikipedia - find when specific text was first added to an article.

**Features:**
- Binary search through revision history to find first appearance of text
- Visual diff showing exact changes made
- Timeline visualization of the search process
- Searches lead section text (before first header)
- Handles "local minimum" (text added, removed, re-added)

**Usage:**
```bash
streamlit run wiki_blame.py
```

**How it works:**
1. Enter Wikipedia page title
2. Enter text to search for (1-2 words typically)
3. Binary search finds first occurrence in specified time range (default: 2 years)
4. Shows the edit, editor, timestamp, and visual diff

### 2. Bias Analyzer (`wiki_bias_analyzer.py`)
Analyzes Wikipedia edit history to detect potential bias patterns using LLM analysis.

**Features:**
- Tracks byte-level changes (+/- for each edit)
- User activity analysis - most active editors and patterns
- LLM-powered analysis using Claude
- Flags suspicious activity (high-frequency editors, large deletions)
- Exports data to CSV and generates reports

**Usage:**
```bash
python wiki_bias_analyzer.py
```

**Configuration:**
Edit the `main()` function to customize:
```python
PAGE_TITLE = "Your_Page_Title"  # Wikipedia page to analyze
MAX_REVISIONS = 500              # Number of revisions to fetch
MIN_USER_EDITS = 3               # Minimum edits to include in analysis
SIZE_CHANGE_THRESHOLD = 200      # Highlight edits larger than this (bytes)
```

**Output:**
- `revisions_data.csv` - Raw revision history with metadata
- `user_statistics.csv` - Per-user aggregated statistics
- `analysis_report.txt` - LLM-powered analysis report

### 3. Language Bias Map (`wiki_language_bias_map.py`)
Compare Wikipedia articles across different language editions to detect bias.

**Features:**
- Fetches same article in multiple languages
- Compares content differences between language editions
- Identifies language-specific biases or perspectives
- Shows which information is present/absent in each version

**Usage:**
```bash
streamlit run wiki_language_bias_map.py
```

### 4. Streamlit Dashboard (`streamlit_app.py`)
Interactive web interface for revision analysis.

**Features:**
- Date range filtering
- User activity visualization
- Edit frequency analysis
- Interactive charts and graphs

**Usage:**
```bash
streamlit run streamlit_app.py
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
For tools using Claude LLM analysis:
```bash
export ANTHROPIC_API_KEY_NC='your-api-key-here'
```

**Note:** The `_NC` suffix avoids conflicts with Claude Code CLI tool.

## Technical Details

### Binary Search Algorithm (Wiki Blame)
- Fetches all revisions in time range (paginated, 500 per request)
- Binary search on revision array (not timestamps!)
- O(log n) complexity where n = number of revisions
- Each checked revision appears on the timeline visualization
- Verifies previous revision doesn't contain text

### API Usage
- Wikipedia MediaWiki API
- Rate limited: ~0.1 seconds between requests
- `rvlimit=500` max revisions per request
- Pagination handled automatically

### Text Extraction
- Extracts lead section (before first `==Header==`)
- Removes wiki markup, templates, references
- Plain text comparison for searching

## Examples

### Find When Text Was Added
```bash
streamlit run wiki_blame.py
# Enter page: "Rashi"
# Enter text: "studied Torah"
# Result: Shows first edit adding this text
```

### Analyze Edit Patterns
```bash
python wiki_bias_analyzer.py
# Analyzes default page or customize in code
# Generates CSV reports and LLM analysis
```

### Compare Language Editions
```bash
streamlit run wiki_language_bias_map.py
# Enter page title
# Select languages to compare
# View differences across editions
```

## Project Structure
```
WikiBiasDetector/
├── wiki_blame.py              # Main wiki-blame tool
├── wiki_bias_analyzer.py      # Bias detection analyzer
├── wiki_language_bias_map.py  # Language comparison tool
├── streamlit_app.py           # Interactive dashboard
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Requirements
- Python 3.8+
- Internet connection (Wikipedia API)
- Anthropic API key (for LLM analysis features)

## License
Educational/Research use

## Contributing
This is a research tool for analyzing Wikipedia edit patterns. Contributions welcome.

## Notes
- "Local minimum": Wiki Blame finds the first occurrence within the search window, but text may have existed earlier (added, removed, re-added)
- Binary search is efficient even for pages with thousands of revisions
- LLM analysis provides insights but should be verified manually
- All data comes from public Wikipedia API
