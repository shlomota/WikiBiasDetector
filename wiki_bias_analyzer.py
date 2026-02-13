"""
Wikipedia Bias Detector - Analyzes edit history to detect potential bias patterns
"""

import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime
import anthropic
import os


class WikipediaAnalyzer:
    """Analyzes Wikipedia page edit history for suspicious patterns"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the analyzer

        Args:
            api_key: Anthropic API key for LLM analysis (defaults to ANTHROPIC_API_KEY_NC env var)
        """
        self.api_endpoint = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikiBiasDetector/1.0 (Educational Research Tool)'
        })

        # Initialize Anthropic client for LLM analysis
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY_NC"))

    def fetch_revisions(self, page_title: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetch revision history for a Wikipedia page

        Args:
            page_title: Title of the Wikipedia page
            limit: Maximum number of revisions to fetch (will paginate if needed)

        Returns:
            DataFrame with revision history including size changes
        """
        print(f"Fetching revisions for: {page_title}")

        revisions = []
        params = {
            'action': 'query',
            'prop': 'revisions',
            'titles': page_title,
            'rvprop': 'ids|timestamp|user|comment|size|userid',  # Key: include 'size' to get byte counts
            'rvlimit': min(500, limit),  # API max is 500 per request
            'format': 'json'
        }

        total_fetched = 0

        while total_fetched < limit:
            response = self.session.get(self.api_endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract page data
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            page_data = pages[page_id]

            if 'revisions' not in page_data:
                print("No revisions found!")
                break

            batch_revisions = page_data['revisions']
            revisions.extend(batch_revisions)
            total_fetched += len(batch_revisions)

            print(f"Fetched {total_fetched} revisions so far...")

            # Check for continuation
            if 'continue' in data and total_fetched < limit:
                params.update(data['continue'])
                time.sleep(0.1)  # Rate limiting
            else:
                break

        # Convert to DataFrame and calculate size changes
        df = pd.DataFrame(revisions)

        if df.empty:
            return df

        # Sort by timestamp (oldest first) to calculate diffs correctly
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate byte change (current size - previous size)
        df['size_change'] = df['size'].diff().fillna(0).astype(int)

        # Add absolute change for filtering
        df['abs_size_change'] = df['size_change'].abs()

        # Reorder columns for readability
        columns_order = ['revid', 'timestamp', 'user', 'userid', 'size', 'size_change',
                        'abs_size_change', 'comment', 'parentid']
        df = df[[col for col in columns_order if col in df.columns]]

        print(f"Total revisions fetched: {len(df)}")
        return df

    def analyze_user_activity(self, df: pd.DataFrame, min_edits: int = 5) -> pd.DataFrame:
        """
        Analyze user activity patterns to identify potentially suspicious users

        Args:
            df: DataFrame with revision history
            min_edits: Minimum number of edits to be included in analysis

        Returns:
            DataFrame with user statistics
        """
        user_stats = df.groupby('user').agg({
            'revid': 'count',  # Number of edits
            'size_change': ['sum', 'mean', 'std'],  # Total, average, and std of changes
            'abs_size_change': 'mean',  # Average absolute change
            'timestamp': ['min', 'max']  # First and last edit
        }).reset_index()

        # Flatten column names
        user_stats.columns = ['user', 'edit_count', 'total_size_change', 'avg_size_change',
                             'std_size_change', 'avg_abs_change', 'first_edit', 'last_edit']

        # Calculate activity span in days
        user_stats['activity_span_days'] = (
            user_stats['last_edit'] - user_stats['first_edit']
        ).dt.total_seconds() / 86400

        # Calculate edits per day (avoid division by zero)
        user_stats['edits_per_day'] = user_stats.apply(
            lambda row: row['edit_count'] / max(row['activity_span_days'], 1), axis=1
        )

        # Filter by minimum edits
        user_stats = user_stats[user_stats['edit_count'] >= min_edits]

        # Sort by edit count (most active first)
        user_stats = user_stats.sort_values('edit_count', ascending=False).reset_index(drop=True)

        return user_stats

    def analyze_summaries_with_llm(self, df: pd.DataFrame, focus: str = "anti-Israel bias") -> str:
        """
        Use LLM to analyze edit summaries for patterns

        Args:
            df: DataFrame with revision history
            focus: What type of bias/pattern to look for

        Returns:
            LLM analysis summary
        """
        # Get top edits by size change (reduced to 12 for cost efficiency)
        top_edits = df.nlargest(12, 'abs_size_change')[['user', 'size_change', 'comment', 'timestamp']]

        # Format for LLM with truncated comments to save tokens
        edits_text = "Significant edits:\n\n"
        for idx, row in top_edits.iterrows():
            comment = str(row['comment'])[:80] + ('...' if len(str(row['comment'])) > 80 else '')
            edits_text += f"- {row['timestamp']:%Y-%m-%d}: {row['user']} ({row['size_change']:+d}b): {comment}\n"

        prompt = f"""Analyze these Wikipedia edits for patterns related to {focus}.

{edits_text}

Identify: 1) Edit type patterns, 2) Systematic user behavior, 3) Potential bias indicators, 4) Coordinated activity.

Be concise and objective. Note legitimate edits vs suspicious patterns."""

        print("Analyzing edit summaries with LLM...")

        message = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,  # Reduced from 2000 to save costs
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text

    def generate_report(self, page_title: str, df: pd.DataFrame, user_stats: pd.DataFrame,
                       llm_analysis: str, min_change_threshold: int = 100) -> str:
        """
        Generate a comprehensive analysis report

        Args:
            page_title: Title of the analyzed page
            df: DataFrame with revision history
            user_stats: DataFrame with user statistics
            llm_analysis: LLM analysis text
            min_change_threshold: Minimum byte change to highlight

        Returns:
            Formatted report string
        """
        report = f"""
{'='*80}
WIKIPEDIA BIAS ANALYSIS REPORT
{'='*80}
Page: {page_title}
Analysis Date: {datetime.now():%Y-%m-%d %H:%M:%S}
Total Revisions Analyzed: {len(df)}
Date Range: {df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}
{'='*80}

OVERALL STATISTICS
{'-'*80}
Total Content Added: +{df[df['size_change'] > 0]['size_change'].sum():,} bytes
Total Content Removed: {df[df['size_change'] < 0]['size_change'].sum():,} bytes
Net Change: {df['size_change'].sum():+,} bytes
Average Edit Size: {df['size_change'].mean():.1f} bytes

TOP 10 MOST ACTIVE USERS
{'-'*80}
"""
        # Add top users table
        for idx, row in user_stats.head(10).iterrows():
            report += f"\n{idx+1}. {row['user']}\n"
            report += f"   Edits: {row['edit_count']} | Net change: {row['total_size_change']:+,} bytes | "
            report += f"Avg: {row['avg_size_change']:+.1f} bytes | Edits/day: {row['edits_per_day']:.2f}\n"

        report += f"\n\nLARGEST EDITS (>{min_change_threshold} bytes change)\n"
        report += f"{'-'*80}\n"

        large_edits = df[df['abs_size_change'] > min_change_threshold].sort_values(
            'abs_size_change', ascending=False
        ).head(15)

        for idx, row in large_edits.iterrows():
            report += f"\n{row['timestamp']:%Y-%m-%d %H:%M} | {row['user']:20s} | "
            report += f"{row['size_change']:+6d} bytes\n"
            report += f"  Comment: {row['comment'][:100]}\n"

        report += f"\n\nLLM ANALYSIS\n"
        report += f"{'-'*80}\n"
        report += llm_analysis

        report += f"\n\n{'='*80}\n"

        return report


def main():
    """Main execution function"""

    # Configuration
    PAGE_TITLE = "October 7 attacks"
    MAX_REVISIONS = 500
    MIN_USER_EDITS = 3
    SIZE_CHANGE_THRESHOLD = 200  # Highlight edits larger than this

    # Initialize analyzer
    analyzer = WikipediaAnalyzer()

    # Fetch revisions
    print("\n" + "="*80)
    print("STEP 1: Fetching revision history")
    print("="*80)
    df = analyzer.fetch_revisions(PAGE_TITLE, limit=MAX_REVISIONS)

    if df.empty:
        print("No revisions found. Exiting.")
        return

    # Save raw data
    df.to_csv('revisions_data.csv', index=False)
    print(f"\n✓ Saved raw data to: revisions_data.csv")

    # Analyze user activity
    print("\n" + "="*80)
    print("STEP 2: Analyzing user activity patterns")
    print("="*80)
    user_stats = analyzer.analyze_user_activity(df, min_edits=MIN_USER_EDITS)
    user_stats.to_csv('user_statistics.csv', index=False)
    print(f"\n✓ Saved user statistics to: user_statistics.csv")
    print(f"✓ Found {len(user_stats)} users with {MIN_USER_EDITS}+ edits")

    # LLM analysis
    print("\n" + "="*80)
    print("STEP 3: LLM-based edit summary analysis")
    print("="*80)
    llm_analysis = analyzer.analyze_summaries_with_llm(df)

    # Generate report
    print("\n" + "="*80)
    print("STEP 4: Generating final report")
    print("="*80)
    report = analyzer.generate_report(
        PAGE_TITLE, df, user_stats, llm_analysis,
        min_change_threshold=SIZE_CHANGE_THRESHOLD
    )

    # Save and display report
    with open('analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n✓ Full report saved to: analysis_report.txt")

    # Show quick stats on potentially suspicious activity
    print("\n" + "="*80)
    print("QUICK INSIGHTS")
    print("="*80)

    # Users with high edit frequency
    high_freq_users = user_stats[user_stats['edits_per_day'] > 2]
    if not high_freq_users.empty:
        print(f"\n⚠ Users with >2 edits/day: {len(high_freq_users)}")
        for _, user in high_freq_users.iterrows():
            print(f"  - {user['user']}: {user['edits_per_day']:.2f} edits/day")

    # Large deletions
    large_deletions = df[df['size_change'] < -500]
    if not large_deletions.empty:
        print(f"\n⚠ Large content removals (>500 bytes): {len(large_deletions)}")
        for _, edit in large_deletions.head(5).iterrows():
            print(f"  - {edit['user']}: {edit['size_change']} bytes on {edit['timestamp']:%Y-%m-%d}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
