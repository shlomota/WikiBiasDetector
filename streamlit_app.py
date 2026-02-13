"""
Wikipedia Bias Detector - Streamlit App
Interactive dashboard for detecting potential anti-Israel bias in Wikipedia edits
"""

import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import anthropic
import os
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed


class WikipediaAnalyzer:
    """Analyzes Wikipedia page edit history for potential bias patterns"""

    def __init__(self, api_key=None):
        self.api_endpoint = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikiBiasDetector/1.0 (Educational Research Tool)'
        })
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY_NC"))

    def fetch_revisions_by_date(self, page_title, start_date, end_date=None):
        """
        Fetch all revisions between start_date and end_date

        Args:
            page_title: Wikipedia page title
            start_date: Start date (datetime object or string YYYY-MM-DD)
            end_date: End date (defaults to now)

        Returns:
            DataFrame with all revisions in date range
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Convert to Wikipedia timestamp format
        start_ts = start_date.strftime("%Y%m%d%H%M%S")
        end_ts = end_date.strftime("%Y%m%d%H%M%S")

        st.info(f"Fetching revisions from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}...")

        revisions = []
        params = {
            'action': 'query',
            'prop': 'revisions',
            'titles': page_title,
            'rvprop': 'ids|timestamp|user|comment|size|userid',
            'rvlimit': 500,
            'rvstart': end_ts,  # Start from most recent
            'rvend': start_ts,  # End at earliest
            'rvdir': 'older',
            'format': 'json'
        }

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_fetched = 0
        while True:
            response = self.session.get(self.api_endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            page_data = pages[page_id]

            if 'revisions' not in page_data:
                break

            batch = page_data['revisions']
            revisions.extend(batch)
            total_fetched += len(batch)

            status_text.text(f"Fetched {total_fetched} revisions...")
            progress_bar.progress(min(total_fetched / 1000, 1.0))  # Assume ~1000 max

            # Check for continuation
            if 'continue' in data:
                params.update(data['continue'])
                time.sleep(0.15)  # Rate limiting
            else:
                break

        progress_bar.empty()
        status_text.empty()

        # Convert to DataFrame
        df = pd.DataFrame(revisions)

        if df.empty:
            return df

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate size changes
        df['size_change'] = df['size'].diff().fillna(0).astype(int)
        df['abs_size_change'] = df['size_change'].abs()

        # Add Wikipedia URLs
        df['diff_url'] = df.apply(
            lambda row: f"https://en.wikipedia.org/w/index.php?diff={row['revid']}&oldid={row.get('parentid', 'prev')}",
            axis=1
        )
        df['diff_current_url'] = df.apply(
            lambda row: f"https://en.wikipedia.org/w/index.php?diff=cur&oldid={row['revid']}",
            axis=1
        )
        df['permalink'] = df.apply(
            lambda row: f"https://en.wikipedia.org/w/index.php?oldid={row['revid']}",
            axis=1
        )
        df['user_page_url'] = df.apply(
            lambda row: f"https://en.wikipedia.org/wiki/User:{row['user'].replace(' ', '_')}",
            axis=1
        )
        df['user_contribs_url'] = df.apply(
            lambda row: f"https://en.wikipedia.org/wiki/Special:Contributions/{row['user'].replace(' ', '_')}",
            axis=1
        )

        st.success(f"✓ Fetched {len(df)} revisions from {df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}")

        return df

    def fetch_diff_content(self, from_revid, to_revid):
        """
        Fetch the actual diff content between two revisions
        Extracts clean ADDED vs REMOVED text

        Returns:
            dict with 'added' and 'removed' text
        """
        try:
            params = {
                'action': 'compare',
                'fromrev': from_revid,
                'torev': to_revid,
                'format': 'json',
                'prop': 'diff'
            }

            response = self.session.get(self.api_endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if 'compare' in data and '*' in data['compare']:
                diff_html = data['compare']['*']
                import re

                # Extract ADDED lines (class="diff-addedline")
                # Look for the full <td class="diff-addedline..."><div>CONTENT</div></td>
                added_matches = re.findall(
                    r'<td class="diff-addedline[^"]*"[^>]*><div>(.*?)</div></td>',
                    diff_html,
                    re.DOTALL
                )

                # Extract REMOVED lines (class="diff-deletedline")
                removed_matches = re.findall(
                    r'<td class="diff-deletedline[^"]*"[^>]*><div>(.*?)</div></td>',
                    diff_html,
                    re.DOTALL
                )

                # Clean HTML tags but keep text
                def clean_html(text):
                    # Remove <ins> and <del> tags but keep their content
                    text = re.sub(r'<ins[^>]*>', '[ADDED:', text)
                    text = re.sub(r'</ins>', ']', text)
                    text = re.sub(r'<del[^>]*>', '[REMOVED:', text)
                    text = re.sub(r'</del>', ']', text)
                    # Remove other HTML tags
                    text = re.sub(r'<[^>]+>', '', text)
                    # Decode entities
                    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
                    return text.strip()

                added_text = '\n'.join(clean_html(m) for m in added_matches if clean_html(m))
                removed_text = '\n'.join(clean_html(m) for m in removed_matches if clean_html(m))

                return {
                    'added': added_text[:2500],  # Limit for LLM cost
                    'removed': removed_text[:2500],
                    'has_content': bool(added_text or removed_text)
                }

            return {'added': '', 'removed': '', 'has_content': False}

        except Exception as e:
            return {'added': '', 'removed': '', 'has_content': False, 'error': str(e)}

    def get_metadata_priority_score(self, row):
        """
        Simple metadata-based priority score to select candidates for LLM analysis
        NOT a bias score - just helps prioritize which edits to send to LLM

        Heavily weights edits to important sections (Top/lead paragraph)

        Returns:
            float: Priority score (higher = should analyze with LLM)
        """
        score = 0
        comment = str(row.get('comment', ''))
        comment_lower = comment.lower()
        size_change = row.get('size_change', 0)

        # CRITICAL: Must have content being ADDED
        if size_change <= 0:
            return 0  # Pure deletions not analyzed

        # Skip obvious maintenance
        maintenance_keywords = ['typo', 'citation bot', 'duplicate', 'formatting', 'fixed reference']
        if any(word in comment_lower for word in maintenance_keywords):
            return 0

        # Prioritize by size of addition
        if size_change > 2000:
            score += 5
        elif size_change > 1000:
            score += 4
        elif size_change > 500:
            score += 3
        elif size_change > 200:
            score += 2
        elif size_change > 100:
            score += 1
        else:
            return 0  # Too small

        # SECTION ANALYSIS - Most important factor!
        # Check for section markers in comment (format: /* Section Name */)
        import re
        section_match = re.search(r'/\*\s*([^*]+?)\s*\*/', comment)

        if section_match:
            section_name = section_match.group(1).lower()

            # Lead paragraph - MAXIMUM importance
            if section_name == 'top':
                score *= 3.0  # Triple the score!

            # Early important sections
            elif any(s in section_name for s in ['background', 'overview', 'description', 'definition', 'summary']):
                score *= 2.0  # Double the score

            # Less important sections - reduce priority
            elif any(s in section_name for s in ['references', 'external links', 'see also', 'notes', 'sources', 'further reading', 'bibliography']):
                score *= 0.3  # Reduce to 30%

            # Works cited / citations
            elif 'works cited' in section_name or 'citations' in section_name:
                score *= 0.2  # Very low priority

        # Boost for edit wars (might indicate controversial content)
        if any(word in comment_lower for word in ['revert', 'undo', 'restore', 'rv ']):
            score += 2

        # Boost for NPOV disputes
        if 'npov' in comment_lower or 'pov' in comment_lower or 'bias' in comment_lower:
            score += 2

        return score

    def score_edit_with_llm(self, row, diff_content):
        """
        Use LLM to score edit bias based on actual content

        Args:
            row: Edit metadata
            diff_content: Dict with 'added' and 'removed' text

        Returns:
            dict: {'bias_score': float, 'explanation': str}
        """
        if not diff_content.get('has_content'):
            return {'bias_score': 0, 'explanation': 'No diff content available'}

        added = diff_content.get('added', '')[:1500]
        removed = diff_content.get('removed', '')[:1500]

        # If nothing added, not suspicious
        if not added or len(added.strip()) < 10:
            return {'bias_score': 0, 'explanation': 'Pure deletion - not suspicious'}

        # Extract section name if present
        import re
        section_info = ""
        section_match = re.search(r'/\*\s*([^*]+?)\s*\*/', str(row.get('comment', '')))
        if section_match:
            section_name = section_match.group(1)
            section_info = f"\nSection edited: **{section_name}**"
            if section_name.lower() == 'top':
                section_info += " (LEAD PARAGRAPH - highest importance!)"
            elif any(s in section_name.lower() for s in ['references', 'external links', 'works cited']):
                section_info += " (Low importance section)"

        prompt = f"""Analyze this Wikipedia edit for anti-Israel bias.

**Edit Details:**
User: {row['user']}
Date: {row['timestamp']:%Y-%m-%d %H:%M}
Size change: {row['size_change']:+d} bytes
Comment: {row['comment']}{section_info}

**Content ADDED:**
{added}

**Content REMOVED:**
{removed}

**Your Task:**
Detect anti-Israel bias in Wikipedia edits. Look for:

1. **Biased sources**: Al Jazeera, Haaretz op-eds, Electronic Intifada, Mondoweiss, fringe NGOs
2. **Fringe/extreme-left scholars**: Ilan Pappé, Norman Finkelstein, Raz Segal, ignoring mainstream scholarship
3. **Hiding Hamas accountability**: Omitting Hamas terrorism, rocket attacks, use of human shields, October 7th massacre
4. **Loaded language**: "Zionist entity", "settler-colonial", inflammatory terms without balance
5. **Logical jumps**: Claims not supported by cited sources, inflated casualty figures
6. **Anti-Zionism/identity erasure**: Denying Jewish connection to Israel, "from river to sea" rhetoric
7. **One-sided narrative**: Presenting only Palestinian perspective, removing Israeli self-defense context
8. **NPOV violations**: Presenting contested claims as facts

**Respond with:**
Score: [0-10, where 0=neutral/pro-balance, 10=extreme anti-Israel bias]
Explanation: [2-3 sentences explaining the score]

Be objective. Many edits are legitimate. Only flag clear bias."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            # Parse score from response
            import re
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response_text)
            if score_match:
                bias_score = float(score_match.group(1))
            else:
                # Try to find just a number at the start
                score_match = re.search(r'^(\d+(?:\.\d+)?)', response_text)
                bias_score = float(score_match.group(1)) if score_match else 5.0

            # Extract explanation (everything after "Explanation:")
            explanation = response_text
            if 'Explanation:' in response_text:
                explanation = response_text.split('Explanation:')[1].strip()

            return {
                'bias_score': min(10, max(0, bias_score)),
                'explanation': explanation
            }

        except Exception as e:
            return {
                'bias_score': 0,
                'explanation': f'Error analyzing: {str(e)}'
            }



def main():
    st.set_page_config(
        page_title="Wikipedia Bias Detector",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Wikipedia Anti-Israel Bias Detector")
    st.markdown("Analyze Wikipedia edit history to detect potential anti-Israel bias patterns")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    page_title = st.sidebar.text_input(
        "Wikipedia Page Title",
        value="October 7 attacks",
        help="Enter the exact Wikipedia page title"
    )

    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2025, 1, 1),
        help="Fetch revisions from this date"
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        help="Fetch revisions until this date"
    )

    min_bias_score = st.sidebar.slider(
        "Minimum Bias Score",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Only show edits with bias score above this threshold"
    )

    top_n_for_llm = st.sidebar.number_input(
        "Top N Edits for LLM Analysis",
        min_value=1,
        max_value=200,
        value=10,
        step=5,
        help="Analyze top N edits with LLM (based on metadata priority). Costs ~$0.01-0.02 per edit."
    )

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = WikipediaAnalyzer()

    # Fetch data button
    if st.sidebar.button("🔄 Fetch & Analyze Revisions", type="primary"):
        with st.spinner("Fetching revisions..."):
            df = st.session_state.analyzer.fetch_revisions_by_date(
                page_title,
                start_date,
                end_date
            )

            if not df.empty:
                # Calculate priority scores (metadata only - for ranking)
                with st.spinner("Calculating priority scores (metadata)..."):
                    df['priority_score'] = df.apply(
                        st.session_state.analyzer.get_metadata_priority_score,
                        axis=1
                    )

                # Initialize LLM scores columns
                df['llm_bias_score'] = None
                df['llm_explanation'] = None

                st.session_state.df = df
                st.success(f"✓ Fetched {len(df)} revisions! Found {len(df[df['priority_score'] > 0])} candidates for analysis.")
            else:
                st.error("No revisions found in the specified date range.")

    # Main content
    if st.session_state.df is not None:
        df = st.session_state.df

        # Get candidates (positive priority score)
        df_candidates = df[df['priority_score'] > 0].copy()
        df_candidates = df_candidates.sort_values('priority_score', ascending=False)

        # Get LLM-analyzed edits
        df_llm_analyzed = df[df['llm_bias_score'].notna()].copy()
        df_llm_analyzed = df_llm_analyzed.sort_values('llm_bias_score', ascending=False)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Revisions", len(df))
        with col2:
            st.metric("Candidates (has additions)", len(df_candidates))
        with col3:
            st.metric("LLM Analyzed", len(df_llm_analyzed))
        with col4:
            if len(df_llm_analyzed) > 0:
                avg_bias = df_llm_analyzed['llm_bias_score'].mean()
                st.metric("Avg Bias Score", f"{avg_bias:.1f}/10")
            else:
                st.metric("Avg Bias Score", "N/A")

        # Button to run LLM analysis
        if len(df_candidates) > 0:
            st.divider()
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                analyze_count = min(top_n_for_llm, len(df_candidates))
                cost_estimate = analyze_count * 0.015  # ~$0.015 per edit
                if st.button(f"🤖 Analyze Top {analyze_count} Edits (~${cost_estimate:.2f})", type="primary", use_container_width=True):
                    with st.spinner(f"Running parallel LLM analysis on top {min(top_n_for_llm, len(df_candidates))} edits..."):
                        top_candidates = df_candidates.head(top_n_for_llm)

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Create analyzer reference for thread workers
                        analyzer = st.session_state.analyzer

                        def analyze_single_edit(idx, row):
                            """Worker function to analyze one edit"""
                            parent_id = row.get('parentid', 0)
                            if not parent_id:
                                return idx, None

                            # Fetch diff
                            diff_content = analyzer.fetch_diff_content(parent_id, row['revid'])

                            # Get LLM score
                            result = analyzer.score_edit_with_llm(row, diff_content)

                            return idx, result

                        # Run in parallel with limited workers (avoid rate limits)
                        results = {}
                        completed = 0

                        with ThreadPoolExecutor(max_workers=5) as executor:
                            # Submit all tasks
                            future_to_idx = {
                                executor.submit(analyze_single_edit, idx, row): idx
                                for idx, row in top_candidates.iterrows()
                            }

                            # Process as they complete
                            for future in as_completed(future_to_idx):
                                idx, result = future.result()

                                if result:
                                    results[idx] = result

                                completed += 1
                                progress_bar.progress(completed / len(top_candidates))
                                status_text.text(f"Analyzed {completed}/{len(top_candidates)} edits")

                        # Update dataframe with results
                        for idx, result in results.items():
                            df.at[idx, 'llm_bias_score'] = result['bias_score']
                            df.at[idx, 'llm_explanation'] = result['explanation']

                        progress_bar.empty()
                        status_text.empty()

                        # Update session state
                        st.session_state.df = df
                        st.success(f"✓ Parallel LLM analysis complete! Analyzed {len(results)} edits.")
                        st.rerun()

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Top 100 Candidates (Metadata)",
            "🎯 LLM Bias Leaderboard",
            "👥 User Analysis",
            "📊 Visualizations"
        ])

        with tab1:
            st.header("Top 100 Candidate Edits (Metadata-Based Priority)")
            st.markdown("Edits with content additions, ranked by size and activity patterns. **These need LLM analysis for bias scoring.**")

            if len(df_candidates) == 0:
                st.warning("No candidate edits found (need edits with positive size changes).")
            else:
                st.info(f"Showing top {min(100, len(df_candidates))} of {len(df_candidates)} candidates")

                # Show top 100
                for i, (idx, row) in enumerate(df_candidates.head(100).iterrows()):
                    # Extract section from comment
                    import re
                    section_badge = ""
                    section_match = re.search(r'/\*\s*([^*]+?)\s*\*/', str(row['comment']))
                    if section_match:
                        section_name = section_match.group(1)
                        if section_name.lower() == 'top':
                            section_badge = "🔴 **TOP**"
                        elif any(s in section_name.lower() for s in ['background', 'overview']):
                            section_badge = f"🟡 *{section_name}*"
                        elif any(s in section_name.lower() for s in ['references', 'external', 'works cited']):
                            section_badge = f"⚪ ~~{section_name}~~"
                        else:
                            section_badge = f"🟢 *{section_name}*"

                    with st.container():
                        col1, col2, col3 = st.columns([2, 3, 2])

                        with col1:
                            st.markdown(f"**#{i+1}** | Priority: {row['priority_score']:.1f}")
                            st.markdown(f"🗓️ {row['timestamp']:%Y-%m-%d %H:%M}")
                            if section_badge:
                                st.markdown(f"📍 {section_badge}")
                            if row['llm_bias_score'] is not None:
                                st.markdown(f"🤖 **LLM Score: {row['llm_bias_score']:.1f}/10**")
                            else:
                                st.markdown(f"⏳ *Not yet analyzed*")

                        with col2:
                            st.markdown(f"**User:** [{row['user']}]({row['user_page_url']})")
                            st.markdown(f"**Size Change:** +{row['size_change']:,} bytes")
                            comment_display = row['comment'][:200]
                            if len(row['comment']) > 200:
                                comment_display += "..."
                            st.markdown(f"**Comment:** {comment_display}")

                        with col3:
                            st.markdown("**Links:**")
                            st.markdown(f"[📝 View Edit]({row['diff_url']})")
                            st.markdown(f"[👤 User Page]({row['user_page_url']})")
                            st.markdown(f"[📜 Contributions]({row['user_contribs_url']})")

                        if row['llm_explanation'] is not None:
                            with st.expander("🤖 LLM Analysis"):
                                st.write(row['llm_explanation'])

                        st.divider()

        with tab2:
            st.header("🎯 LLM Bias Leaderboard")
            st.markdown("Edits analyzed by AI and scored for anti-Israel bias (0=neutral, 10=extreme bias)")

            if len(df_llm_analyzed) == 0:
                st.warning("No edits have been analyzed yet. Click the 'Analyze with LLM' button above to start.")
            else:
                st.success(f"Showing {len(df_llm_analyzed)} LLM-analyzed edits, sorted by bias score")

                # Filter by minimum score if desired
                min_llm_score = st.slider("Minimum LLM Bias Score", 0.0, 10.0, 3.0, 0.5)
                df_llm_filtered = df_llm_analyzed[df_llm_analyzed['llm_bias_score'] >= min_llm_score]

                st.info(f"{len(df_llm_filtered)} edits with score ≥ {min_llm_score}")

                for i, (idx, row) in enumerate(df_llm_filtered.iterrows()):
                    # Extract section from comment
                    import re
                    section_badge = ""
                    section_match = re.search(r'/\*\s*([^*]+?)\s*\*/', str(row['comment']))
                    if section_match:
                        section_name = section_match.group(1)
                        if section_name.lower() == 'top':
                            section_badge = "🔴 **TOP**"
                        elif any(s in section_name.lower() for s in ['background', 'overview']):
                            section_badge = f"🟡 *{section_name}*"
                        elif any(s in section_name.lower() for s in ['references', 'external', 'works cited']):
                            section_badge = f"⚪ ~~{section_name}~~"
                        else:
                            section_badge = f"🟢 *{section_name}*"

                    with st.container():
                        col1, col2, col3 = st.columns([2, 3, 2])

                        with col1:
                            st.markdown(f"**#{i+1}**")
                            st.markdown(f"🗓️ {row['timestamp']:%Y-%m-%d %H:%M}")
                            if section_badge:
                                st.markdown(f"📍 {section_badge}")
                            bias_score = row['llm_bias_score']
                            if bias_score >= 7:
                                st.markdown(f"🔴 **Bias Score: {bias_score:.1f}/10**")
                            elif bias_score >= 4:
                                st.markdown(f"🟡 **Bias Score: {bias_score:.1f}/10**")
                            else:
                                st.markdown(f"🟢 **Bias Score: {bias_score:.1f}/10**")

                        with col2:
                            st.markdown(f"**User:** [{row['user']}]({row['user_page_url']})")
                            st.markdown(f"**Size Change:** +{row['size_change']:,} bytes")
                            comment_display = row['comment'][:200]
                            if len(row['comment']) > 200:
                                comment_display += "..."
                            st.markdown(f"**Comment:** {comment_display}")

                        with col3:
                            st.markdown("**Links:**")
                            st.markdown(f"[📝 View Edit]({row['diff_url']})")
                            st.markdown(f"[👤 User Page]({row['user_page_url']})")
                            st.markdown(f"[📜 Contributions]({row['user_contribs_url']})")

                        with st.expander("🤖 LLM Analysis", expanded=(bias_score >= 7)):
                            st.write(row['llm_explanation'])

                        st.divider()

        with tab3:
            st.header("User Activity Analysis")

            # Filter to users with candidates or LLM scores
            active_df = df[(df['priority_score'] > 0) | (df['llm_bias_score'].notna())]

            if len(active_df) == 0:
                st.warning("No user activity to analyze yet.")
            else:
                # Aggregate user stats
                user_stats = active_df.groupby('user').agg({
                    'revid': 'count',
                    'size_change': ['sum', 'mean'],
                    'priority_score': ['mean', 'max'],
                    'llm_bias_score': ['mean', 'max', 'count'],
                    'timestamp': ['min', 'max']
                }).reset_index()

                user_stats.columns = ['user', 'edit_count', 'total_size_change', 'avg_size_change',
                                     'avg_priority', 'max_priority', 'avg_llm_score', 'max_llm_score',
                                     'llm_analyzed_count', 'first_edit', 'last_edit']

                # Sort by average LLM score (if available), otherwise by priority
                if user_stats['llm_analyzed_count'].sum() > 0:
                    user_stats = user_stats.sort_values('avg_llm_score', ascending=False, na_position='last')
                else:
                    user_stats = user_stats.sort_values('avg_priority', ascending=False)

                st.subheader("Most Suspicious Users")

                for idx, row in user_stats.head(15).iterrows():
                    # Get user page URL
                    user_page = f"https://en.wikipedia.org/wiki/User:{row['user'].replace(' ', '_')}"
                    user_contribs = f"https://en.wikipedia.org/wiki/Special:Contributions/{row['user'].replace(' ', '_')}"

                    llm_analyzed = row['llm_analyzed_count'] if not pd.isna(row['llm_analyzed_count']) else 0

                    with st.expander(f"**[{row['user']}]({user_page})** - {int(llm_analyzed)} LLM analyzed"):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Total Edits", int(row['edit_count']))
                            st.metric("LLM Analyzed", int(llm_analyzed))

                        with col2:
                            if not pd.isna(row['avg_llm_score']):
                                st.metric("Avg LLM Bias Score", f"{row['avg_llm_score']:.1f}/10")
                                st.metric("Max LLM Bias Score", f"{row['max_llm_score']:.1f}/10")
                            else:
                                st.metric("Avg Priority", f"{row['avg_priority']:.1f}")
                                st.metric("Max Priority", f"{row['max_priority']:.1f}")

                        with col3:
                            st.metric("Total Added", f"+{int(row['total_size_change']):,} bytes")
                            st.metric("Avg Addition", f"+{int(row['avg_size_change']):,} bytes")

                        st.write(f"📅 Active: {row['first_edit']:%Y-%m-%d} to {row['last_edit']:%Y-%m-%d}")
                        st.markdown(f"[👤 User Page]({user_page}) | [📜 All Contributions]({user_contribs})")

                        # Show user's top edits
                        user_edits = active_df[active_df['user'] == row['user']]
                        if user_edits['llm_bias_score'].notna().any():
                            user_edits = user_edits.sort_values('llm_bias_score', ascending=False, na_position='last')
                        else:
                            user_edits = user_edits.sort_values('priority_score', ascending=False)

                        st.markdown("**Top Edits:**")
                        for _, edit in user_edits.head(5).iterrows():
                            if not pd.isna(edit['llm_bias_score']):
                                st.markdown(
                                    f"- {edit['timestamp']:%Y-%m-%d}: +{edit['size_change']} bytes "
                                    f"(**LLM: {edit['llm_bias_score']:.1f}/10**) - "
                                    f"[View]({edit['diff_url']})"
                                )
                            else:
                                st.markdown(
                                    f"- {edit['timestamp']:%Y-%m-%d}: +{edit['size_change']} bytes "
                                    f"(priority: {edit['priority_score']:.1f}) - "
                                    f"[View]({edit['diff_url']})"
                                )

        with tab4:
            st.header("Data Visualizations")

            # Timeline of edits - show all with additions
            vis_df = df[df['size_change'] > 0].copy()

            if len(vis_df) > 0:
                # Use LLM score if available, otherwise priority
                vis_df['display_score'] = vis_df['llm_bias_score'].combine_first(vis_df['priority_score'])

                fig_timeline = px.scatter(
                    vis_df,
                    x='timestamp',
                    y='size_change',
                    color='display_score',
                    size='size_change',
                    hover_data=['user', 'comment'],
                    title="Edit Timeline (additions only, colored by bias/priority score)",
                    color_continuous_scale='Reds',
                    labels={'display_score': 'Bias/Priority Score'}
                )
                st.plotly_chart(fig_timeline, width='stretch')

            # Top users by additions
            users_with_additions = df[df['size_change'] > 0]['user'].value_counts().head(15)
            fig_users = px.bar(
                x=users_with_additions.index,
                y=users_with_additions.values,
                labels={'x': 'User', 'y': 'Edits with Additions'},
                title="Top 15 Users by Number of Edits with Additions"
            )
            st.plotly_chart(fig_users, width='stretch')

            # LLM Bias score distribution (if available)
            if len(df_llm_analyzed) > 0:
                fig_bias = px.histogram(
                    df_llm_analyzed,
                    x='llm_bias_score',
                    nbins=20,
                    title="LLM Bias Score Distribution",
                    labels={'llm_bias_score': 'LLM Bias Score (0-10)', 'count': 'Number of Edits'}
                )
                st.plotly_chart(fig_bias, width='stretch')

            # Priority score distribution
            fig_priority = px.histogram(
                df[df['priority_score'] > 0],
                x='priority_score',
                nbins=20,
                title="Priority Score Distribution (Metadata-Based)",
                labels={'priority_score': 'Priority Score', 'count': 'Number of Edits'}
            )
            st.plotly_chart(fig_priority, width='stretch')

    else:
        st.info("👈 Configure settings in the sidebar and click 'Fetch & Analyze Revisions' to begin")

        # Show example
        st.markdown("""
        ### How it works:

        1. **Fetch revisions** from Wikipedia for a specific page and date range
        2. **Calculate bias scores** based on:
           - Large additions/deletions
           - Edit war patterns (reverts, undos)
           - Keywords suggesting bias
           - NPOV disputes
        3. **View leaderboard** of most suspicious edits with direct links to Wikipedia diffs
        4. **Analyze users** to identify patterns of potentially biased editing
        5. **AI analysis** for deep-dive investigation of specific edits

        ### Getting Started:
        - Default page: "October 7 attacks"
        - Default date range: 2025-01-01 to today
        - Adjust bias score threshold to filter results
        """)


if __name__ == "__main__":
    main()
