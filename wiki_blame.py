"""
Wiki Blame - Find when specific text was first added to a Wikipedia article
Similar to git blame, but for Wikipedia edits
"""

import streamlit as st
import requests
from datetime import datetime, timedelta
import time
import re
import difflib
import plotly.graph_objects as go


class WikiBlame:
    """Find when specific text was first introduced to a Wikipedia article"""

    def __init__(self):
        self.api_endpoint = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikiBlame/1.0 (Educational Research Tool)'
        })

    def get_current_content(self, page_title, as_html=False):
        """Get the current version of a page's first section"""
        params = {
            'action': 'query',
            'prop': 'extracts',
            'titles': page_title,
            'exintro': True,  # Get intro section only
            'format': 'json'
        }

        if not as_html:
            params['explaintext'] = True  # Plain text for searching

        response = self.session.get(self.api_endpoint, params=params)
        data = response.json()

        pages = data.get('query', {}).get('pages', {})
        page = list(pages.values())[0]

        if 'extract' in page:
            return page['extract']
        return None

    def get_revision_at_timestamp(self, page_title, timestamp):
        """Get the content of a specific revision by timestamp"""
        params = {
            'action': 'query',
            'prop': 'revisions',
            'titles': page_title,
            'rvprop': 'content|timestamp|user|comment|ids',
            'rvlimit': 1,
            'rvstart': timestamp,
            'rvdir': 'older',
            'format': 'json',
            'rvslots': 'main'
        }

        response = self.session.get(self.api_endpoint, params=params)
        data = response.json()

        pages = data.get('query', {}).get('pages', {})
        page = list(pages.values())[0]

        if 'revisions' in page and len(page['revisions']) > 0:
            rev = page['revisions'][0]
            content = rev.get('slots', {}).get('main', {}).get('*', '')
            return {
                'content': content,
                'timestamp': rev.get('timestamp'),
                'user': rev.get('user'),
                'comment': rev.get('comment', ''),
                'revid': rev.get('revid')
            }
        return None

    def extract_first_section_text(self, wikitext, keep_refs=False):
        """Extract plain text from the first section of wikitext"""
        lines = wikitext.split('\n')
        first_section = []

        for line in lines:
            if line.startswith('==') and '==' in line[2:]:
                break
            first_section.append(line)

        text = '\n'.join(first_section)

        # Remove wiki markup (basic cleaning)
        text = text.replace("'''", '')
        text = text.replace("''", '')
        # Handle wiki links [[link|text]] or [[link]]
        text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', text)
        # Remove templates
        text = re.sub(r'\{\{[^\}]+\}\}', '', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        if not keep_refs:
            # Remove references like [1], [2]
            text = re.sub(r'\[[0-9]+\]', '', text)

        return text.strip()

    def get_previous_revision(self, page_title, revid):
        """Get the previous revision before the given revid"""
        try:
            # Get parent revision ID
            params_parent = {
                'action': 'query',
                'revids': revid,
                'prop': 'revisions',
                'rvprop': 'ids',
                'format': 'json'
            }

            response = self.session.get(self.api_endpoint, params=params_parent)
            data = response.json()
            pages = data.get('query', {}).get('pages', {})

            if not pages:
                return None

            page = list(pages.values())[0]

            if 'revisions' in page and len(page['revisions']) > 0:
                parent_id = page['revisions'][0].get('parentid')
                if parent_id:
                    params_content = {
                        'action': 'query',
                        'revids': parent_id,
                        'prop': 'revisions',
                        'rvprop': 'content',
                        'rvslots': 'main',
                        'format': 'json'
                    }
                    response = self.session.get(self.api_endpoint, params=params_content)
                    data = response.json()
                    pages = data.get('query', {}).get('pages', {})

                    if not pages:
                        return None

                    page = list(pages.values())[0]

                    if 'revisions' in page and len(page['revisions']) > 0:
                        content = page['revisions'][0].get('slots', {}).get('main', {}).get('*', '')
                        return content
        except (KeyError, IndexError, TypeError) as e:
            return None

        return None

    def create_visual_diff(self, old_text, new_text):
        """Create a visual side-by-side diff with highlighting"""
        # Split into sentences for better granularity
        old_sentences = re.split(r'([.!?]\s+)', old_text)
        new_sentences = re.split(r'([.!?]\s+)', new_text)

        # Recombine sentences with their punctuation
        old_parts = []
        for i in range(0, len(old_sentences)-1, 2):
            if i+1 < len(old_sentences):
                old_parts.append(old_sentences[i] + old_sentences[i+1])
            else:
                old_parts.append(old_sentences[i])
        if len(old_sentences) % 2 == 1:
            old_parts.append(old_sentences[-1])

        new_parts = []
        for i in range(0, len(new_sentences)-1, 2):
            if i+1 < len(new_sentences):
                new_parts.append(new_sentences[i] + new_sentences[i+1])
            else:
                new_parts.append(new_sentences[i])
        if len(new_sentences) % 2 == 1:
            new_parts.append(new_sentences[-1])

        # Use difflib to find differences
        matcher = difflib.SequenceMatcher(None, old_parts, new_parts)

        old_html = []
        new_html = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for part in old_parts[i1:i2]:
                    old_html.append(part)
                for part in new_parts[j1:j2]:
                    new_html.append(part)
            elif tag == 'delete':
                for part in old_parts[i1:i2]:
                    old_html.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{part}</span>')
            elif tag == 'insert':
                for part in new_parts[j1:j2]:
                    new_html.append(f'<span style="background-color: #ccffcc;">{part}</span>')
            elif tag == 'replace':
                # Word-level diff for replaced sections
                old_section = ' '.join(old_parts[i1:i2])
                new_section = ' '.join(new_parts[j1:j2])

                # Try word-level comparison
                old_words = old_section.split()
                new_words = new_section.split()

                word_matcher = difflib.SequenceMatcher(None, old_words, new_words)

                old_word_html = []
                new_word_html = []

                for wtag, wi1, wi2, wj1, wj2 in word_matcher.get_opcodes():
                    if wtag == 'equal':
                        old_word_html.extend(old_words[wi1:wi2])
                        new_word_html.extend(new_words[wj1:wj2])
                    elif wtag == 'delete':
                        for word in old_words[wi1:wi2]:
                            old_word_html.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{word}</span>')
                    elif wtag == 'insert':
                        for word in new_words[wj1:wj2]:
                            new_word_html.append(f'<span style="background-color: #ccffcc;">{word}</span>')
                    elif wtag == 'replace':
                        for word in old_words[wi1:wi2]:
                            old_word_html.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{word}</span>')
                        for word in new_words[wj1:wj2]:
                            new_word_html.append(f'<span style="background-color: #ccffcc;">{word}</span>')

                old_html.append(' '.join(old_word_html))
                new_html.append(' '.join(new_word_html))

        return ''.join(old_html), ''.join(new_html)

    def get_revision_list(self, page_title, start_date, end_date):
        """Get list of all revision IDs in a time range"""
        start_ts = start_date.strftime("%Y%m%d%H%M%S")
        end_ts = end_date.strftime("%Y%m%d%H%M%S")

        revisions = []
        continue_token = None

        # Keep fetching until we cover the entire time range or run out of revisions
        while True:
            params = {
                'action': 'query',
                'titles': page_title,
                'prop': 'revisions',
                'rvprop': 'ids|timestamp|user|comment',
                'rvstart': end_ts,
                'rvend': start_ts,
                'rvdir': 'older',
                'rvlimit': 500,  # Max allowed by API
                'format': 'json'
            }

            if continue_token:
                params['rvcontinue'] = continue_token

            response = self.session.get(self.api_endpoint, params=params)
            data = response.json()

            pages = data.get('query', {}).get('pages', {})
            if not pages:
                break

            page = list(pages.values())[0]
            if 'revisions' not in page:
                break

            batch = page['revisions']
            revisions.extend(batch)

            # Check if we've covered the requested time range
            # Revisions are newest-first, so check the last (oldest) revision in this batch
            if batch:
                oldest_in_batch = batch[-1]['timestamp']
                oldest_time = datetime.strptime(oldest_in_batch, "%Y-%m-%dT%H:%M:%SZ")
                if oldest_time <= start_date:
                    # We've reached or passed the start date
                    break

            # Check for continuation
            if 'continue' in data:
                continue_token = data['continue'].get('rvcontinue')
            else:
                # No more revisions available
                break

            time.sleep(0.1)  # Rate limiting

        return revisions

    def binary_search_first_appearance(self, page_title, search_text, start_date, end_date):
        """
        Use binary search to find when the text first appeared

        Args:
            page_title: Wikipedia page title
            search_text: Text to search for
            start_date: Start date (datetime)
            end_date: End date (datetime)

        Returns:
            Dictionary with revision info or None
        """

        # Get list of all revisions in the time range
        search_log = []
        search_timeline = []

        search_log.append(f"Fetching revision list...")
        revision_list = self.get_revision_list(page_title, start_date, end_date)

        if not revision_list:
            return {'error': 'No revisions found in time range'}

        search_log.append(f"Found {len(revision_list)} revisions to search")

        # Check if text exists in most recent revision
        most_recent_revid = revision_list[0]['revid']
        params = {
            'action': 'query',
            'revids': most_recent_revid,
            'prop': 'revisions',
            'rvprop': 'content',
            'rvslots': 'main',
            'format': 'json'
        }
        response = self.session.get(self.api_endpoint, params=params)
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        if not pages:
            return None

        page = list(pages.values())[0]
        if 'revisions' not in page:
            return None

        content = page['revisions'][0].get('slots', {}).get('main', {}).get('*', '')
        text = self.extract_first_section_text(content)

        if search_text not in text:
            return {'error': 'Text not found in current version'}

        # Binary search on revision list (indices)
        left_idx = 0
        right_idx = len(revision_list) - 1
        last_found_idx = 0
        last_found_content = content

        iterations = 0
        max_iterations = 30  # Higher limit since we're searching actual revisions

        while left_idx <= right_idx and iterations < max_iterations:
            iterations += 1

            # Calculate midpoint index
            mid_idx = (left_idx + right_idx) // 2
            mid_rev_info = revision_list[mid_idx]
            mid_revid = mid_rev_info['revid']
            mid_timestamp = datetime.strptime(mid_rev_info['timestamp'], "%Y-%m-%dT%H:%M:%SZ")

            search_log.append(f"Iteration {iterations}: Checking revision {mid_revid} ({mid_timestamp.strftime('%Y-%m-%d %H:%M')})")

            # Get revision content
            params = {
                'action': 'query',
                'revids': mid_revid,
                'prop': 'revisions',
                'rvprop': 'content',
                'rvslots': 'main',
                'format': 'json'
            }
            response = self.session.get(self.api_endpoint, params=params)
            data = response.json()
            pages = data.get('query', {}).get('pages', {})

            if not pages:
                # Skip this revision
                left_idx = mid_idx + 1
                continue

            page = list(pages.values())[0]
            if 'revisions' not in page:
                left_idx = mid_idx + 1
                continue

            content = page['revisions'][0].get('slots', {}).get('main', {}).get('*', '')
            mid_text = self.extract_first_section_text(content)

            if search_text in mid_text:
                # Text found! First appearance is at mid_idx or later (higher index = older)
                last_found_idx = mid_idx
                last_found_content = content
                left_idx = mid_idx + 1  # Search older revisions (higher indices)
                search_log.append(f"  → Text found! Searching earlier (older revisions)...")
                search_timeline.append({
                    'timestamp': mid_timestamp,
                    'found': True,
                    'iteration': iterations,
                    'user': mid_rev_info['user'],
                    'comment': mid_rev_info.get('comment', '')[:100] if mid_rev_info.get('comment') else 'No comment'
                })
            else:
                # Text not found. First appearance is in newer revisions (lower indices)
                right_idx = mid_idx - 1  # Search newer revisions (lower indices)
                search_log.append(f"  → Text not found. Searching later (newer revisions)...")
                search_timeline.append({
                    'timestamp': mid_timestamp,
                    'found': False,
                    'iteration': iterations,
                    'user': mid_rev_info['user'],
                    'comment': mid_rev_info.get('comment', '')[:100] if mid_rev_info.get('comment') else 'No comment'
                })

            time.sleep(0.1)  # Rate limiting

        # Build the result from the found revision
        found_rev_info = revision_list[last_found_idx]

        result_revision = {
            'content': last_found_content,
            'timestamp': found_rev_info['timestamp'],
            'user': found_rev_info['user'],
            'comment': found_rev_info.get('comment', ''),
            'revid': found_rev_info['revid']
        }

        # Final verification and extension if needed
        search_log.append(f"\nFinal verification:")

        # If we found text at the END of our revision list, we need to search further back
        if last_found_idx >= len(revision_list) - 5:  # Near the end
            search_log.append(f"  → Text found near end of revision list, checking if it exists earlier...")

            # Get the previous revision using the parent API
            oldest_revid = revision_list[-1]['revid']
            prev_content = self.get_previous_revision(page_title, oldest_revid)

            if prev_content:
                prev_text = self.extract_first_section_text(prev_content)
                if search_text in prev_text:
                    search_log.append(f"  → ⚠️ Text exists before search range! Consider expanding search window.")
                    search_log.append(f"  → Returning earliest revision in searched range (local minimum).")
                else:
                    search_log.append(f"  → ✓ Text NOT in earlier revisions. Found first occurrence!")
            else:
                search_log.append(f"  → No earlier revisions exist (beginning of page history).")

        elif last_found_idx + 1 < len(revision_list):
            # Check next revision in our list (earlier in time, higher index)
            next_rev_info = revision_list[last_found_idx + 1]
            params = {
                'action': 'query',
                'revids': next_rev_info['revid'],
                'prop': 'revisions',
                'rvprop': 'content',
                'rvslots': 'main',
                'format': 'json'
            }
            response = self.session.get(self.api_endpoint, params=params)
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            if pages:
                page = list(pages.values())[0]
                if 'revisions' in page:
                    next_content = page['revisions'][0].get('slots', {}).get('main', {}).get('*', '')
                    next_text = self.extract_first_section_text(next_content)
                    if search_text in next_text:
                        search_log.append(f"  → ⚠️ Text also in previous revision! This shouldn't happen.")
                    else:
                        search_log.append(f"  → ✓ Text NOT in previous revision. Verified!")
        else:
            search_log.append(f"  → At end of search range.")

        return {
            'revision': result_revision,
            'iterations': iterations,
            'search_log': search_log,
            'timeline': search_timeline
        }

    def get_revision_diff_html(self, page_title, revid, inline=False):
        """Get the HTML diff for a specific revision"""
        params = {
            'action': 'compare',
            'fromrev': revid,
            'torelative': 'prev',
            'format': 'json'
        }

        if inline:
            params['difftype'] = 'inline'

        response = self.session.get(self.api_endpoint, params=params)
        data = response.json()

        if 'compare' in data and '*' in data['compare']:
            return data['compare']['*']
        return None

    def get_rendered_html(self, page_title, revid):
        """Get the rendered HTML for a specific revision using visualeditor API"""
        params = {
            'action': 'visualeditor',
            'paction': 'parse',
            'page': page_title,
            'oldid': revid,
            'format': 'json',
            'formatversion': 2
        }

        response = self.session.get(self.api_endpoint, params=params)
        data = response.json()

        if 'visualeditor' in data and 'content' in data['visualeditor']:
            return data['visualeditor']['content']
        return None

    def extract_first_paragraph_html(self, html_content):
        """Extract all paragraphs from the lead section (before first section header)"""
        from html.parser import HTMLParser

        class LeadSectionExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_body = False
                self.in_paragraph = False
                self.current_para = []
                self.all_paragraphs = []
                self.depth = 0
                self.skip_paragraph = False
                self.hit_section_header = False

            def handle_starttag(self, tag, attrs):
                # Check if we've hit a section header (h2) - stop collecting after this
                if tag == 'h2':
                    self.hit_section_header = True
                    return

                if tag == 'body':
                    self.in_body = True
                elif self.in_body and tag == 'p' and not self.hit_section_header:
                    # Check if this paragraph is inside an infobox or template
                    attrs_dict = dict(attrs)
                    class_attr = attrs_dict.get('class', '')

                    # Skip paragraphs in infoboxes, side boxes, or templates
                    skip_classes = ['infobox', 'sidebar', 'navbox', 'metadata', 'ambox', 'mbox']
                    if any(skip_class in class_attr for skip_class in skip_classes):
                        self.skip_paragraph = True
                    else:
                        self.skip_paragraph = False
                        self.in_paragraph = True
                        self.depth = 1
                elif self.in_paragraph:
                    self.depth += 1
                    attrs_str = ' '.join([f'{k}="{v}"' for k, v in attrs])
                    self.current_para.append(f'<{tag} {attrs_str}>' if attrs_str else f'<{tag}>')

            def handle_endtag(self, tag):
                if self.in_paragraph:
                    self.depth -= 1
                    if tag == 'p' and self.depth == 0:
                        # Check if this paragraph has actual text content
                        para_html = ''.join(self.current_para)
                        # Extract text only to check
                        para_text = re.sub(r'<[^>]+>', '', para_html).strip()

                        # Add paragraph if it has substantial content
                        if para_text and len(para_text) > 50:
                            # Add paragraph separator if we already have content
                            if self.all_paragraphs:
                                self.all_paragraphs.append('\n\n')
                            self.all_paragraphs.extend(self.current_para)

                        self.in_paragraph = False
                        self.current_para = []
                    else:
                        self.current_para.append(f'</{tag}>')

            def handle_data(self, data):
                if self.in_paragraph and not self.skip_paragraph:
                    self.current_para.append(data)

        parser = LeadSectionExtractor()
        parser.feed(html_content)
        return ''.join(parser.all_paragraphs)


def main():
    st.set_page_config(page_title="Wiki Blame - Git Blame for Wiki Edits", page_icon="🔍", layout="wide")

    st.title("🔍 Wiki Blame - Git Blame for Wiki Edits")
    st.subheader("Find the edit responsible for the text you see today.")
    st.markdown("""
    **Note:** This tool searches for the first appearance of text in the last 2 years by default.
    The text might have been added earlier, removed, and re-added later.
    """)

    # Initialize
    if 'wiki_blame' not in st.session_state:
        st.session_state.wiki_blame = WikiBlame()

    wiki = st.session_state.wiki_blame

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        page_title = st.text_input("Wikipedia Page Title", value="October 7 attacks",
                                    help="Enter the exact Wikipedia page title")

    with col2:
        years_back = st.number_input("Years back to search", min_value=1, max_value=10, value=2)

    if st.button("Load Page", type="primary"):
        with st.spinner("Loading current version..."):
            content_text = wiki.get_current_content(page_title, as_html=False)
            content_html = wiki.get_current_content(page_title, as_html=True)
            if content_text and content_html:
                st.session_state.current_content = content_text
                st.session_state.current_content_html = content_html
                st.session_state.page_title = page_title
                st.session_state.years_back = years_back
                st.success("Page loaded!")
            else:
                st.error("Could not load page. Please check the title.")

    # Display current content
    if 'current_content' in st.session_state:
        st.subheader("Current First Paragraph")
        # Display HTML version for better readability
        wiki_style_css = """
        <style>
        .wiki-content {
            font-family: sans-serif;
            line-height: 1.6;
            padding: 15px;
            background-color: #f8f9fa;
            border: 1px solid #a2a9b1;
            border-radius: 5px;
        }
        .wiki-content a {
            color: #0645ad;
            text-decoration: none;
        }
        .wiki-content a:hover {
            text-decoration: underline;
        }
        .wiki-content p {
            margin-bottom: 10px;
        }
        </style>
        """
        st.markdown(wiki_style_css, unsafe_allow_html=True)
        st.markdown(f'<div class="wiki-content">{st.session_state.current_content_html}</div>', unsafe_allow_html=True)

        # Text selection
        st.subheader("Search for Text")
        search_text = st.text_input("Enter the text to find (1-2 words typically)",
                                     help="Enter the exact text you want to find the origin of")

        if st.button("🔎 Find When Text Was Added"):
            if not search_text:
                st.error("Please enter text to search for")
            else:
                with st.spinner("Searching through revisions using binary search..."):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365 * st.session_state.years_back)

                    result = wiki.binary_search_first_appearance(
                        st.session_state.page_title,
                        search_text,
                        start_date,
                        end_date
                    )

                    if result and 'error' not in result:
                        st.session_state.search_result = result
                        st.session_state.search_text = search_text
                    elif result and 'error' in result:
                        st.error(result['error'])
                    else:
                        st.error("Could not find the text in the specified time range")

    # Display results
    if 'search_result' in st.session_state:
        result = st.session_state.search_result
        rev = result['revision']

        st.success("✅ Found the first appearance!")

        # Results in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Date", datetime.strptime(rev['timestamp'], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M"))

        with col2:
            st.markdown("**Editor**")
            user_url = f"https://en.wikipedia.org/wiki/User:{rev['user']}"
            st.markdown(f"[{rev['user']}]({user_url})")

        with col3:
            st.metric("Revisions Checked", result['iterations'])

        # Edit details
        st.subheader("Edit Details")
        st.write(f"**Edit Comment:** {rev['comment'] if rev['comment'] else 'No comment'}")
        st.write(f"**Revision ID:** {rev['revid']}")

        # Link to Wikipedia
        wiki_url = f"https://en.wikipedia.org/w/index.php?diff={rev['revid']}&oldid=prev"
        st.markdown(f"[🔗 View this edit on Wikipedia]({wiki_url})")

        # Binary search timeline visualization
        st.subheader("Binary Search Timeline")
        if 'timeline' in result and result['timeline']:
            timeline = result['timeline']

            # Separate found and not found
            found_entries = [entry for entry in timeline if entry['found']]
            not_found_entries = [entry for entry in timeline if not entry['found']]

            # Create plotly figure
            fig = go.Figure()

            # Calculate max iteration for sizing
            max_iteration = max([entry['iteration'] for entry in timeline])

            # Add trace for text found (green)
            if found_entries:
                found_timestamps = [entry['timestamp'] for entry in found_entries]
                found_iterations = [entry['iteration'] for entry in found_entries]
                found_users = [entry['user'] for entry in found_entries]
                found_comments = [entry['comment'] for entry in found_entries]

                # Calculate sizes: grow as we get closer to the final result
                # Base size 10, grows to 25 for final result
                found_sizes = [10 + (15 * (i / max_iteration)) for i in found_iterations]
                # Make the last found entry (the result) extra large
                last_found_idx = found_iterations.index(max(found_iterations))
                found_sizes[last_found_idx] = 30

                fig.add_trace(go.Scatter(
                    x=found_timestamps,
                    y=found_iterations,
                    mode='markers+text',
                    marker=dict(
                        size=found_sizes,
                        color='#28a745',
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    text=[f"#{i}" for i in found_iterations],
                    textposition="middle right",
                    name='Text Found',
                    customdata=list(zip(found_iterations, found_users, found_comments)),
                    hovertemplate='<b style="color: #28a745;">✓ Text Found</b><br>' +
                                  'Iteration: #%{customdata[0]}<br>' +
                                  'Date: %{x|%Y-%m-%d %H:%M}<br>' +
                                  'Editor: %{customdata[1]}<br>' +
                                  'Comment: %{customdata[2]}<extra></extra>'
                ))

            # Add trace for text not found (red)
            if not_found_entries:
                not_found_timestamps = [entry['timestamp'] for entry in not_found_entries]
                not_found_iterations = [entry['iteration'] for entry in not_found_entries]
                not_found_users = [entry['user'] for entry in not_found_entries]
                not_found_comments = [entry['comment'] for entry in not_found_entries]

                # Calculate sizes: grow as we get closer to convergence
                not_found_sizes = [10 + (15 * (i / max_iteration)) for i in not_found_iterations]

                fig.add_trace(go.Scatter(
                    x=not_found_timestamps,
                    y=not_found_iterations,
                    mode='markers+text',
                    marker=dict(
                        size=not_found_sizes,
                        color='#dc3545',
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    text=[f"#{i}" for i in not_found_iterations],
                    textposition="middle left",
                    name='Text Not Found',
                    customdata=list(zip(not_found_iterations, not_found_users, not_found_comments)),
                    hovertemplate='<b style="color: #dc3545;">✗ Text Not Found</b><br>' +
                                  'Iteration: #%{customdata[0]}<br>' +
                                  'Date: %{x|%Y-%m-%d %H:%M}<br>' +
                                  'Editor: %{customdata[1]}<br>' +
                                  'Comment: %{customdata[2]}<extra></extra>'
                ))

            # Update layout
            fig.update_layout(
                showlegend=True,
                height=400,
                yaxis=dict(
                    title="Iteration #",
                    showticklabels=True,
                    showgrid=True,
                    gridcolor='#e0e0e0',
                    zeroline=False,
                    range=[0, max_iteration + 1],
                    dtick=1  # Show every iteration number
                ),
                xaxis=dict(
                    title="Timeline",
                    showgrid=True,
                    gridcolor='#e0e0e0'
                ),
                plot_bgcolor='white',
                margin=dict(l=60, r=20, t=40, b=60),
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **How to read this:** Each dot represents a revision checked during binary search.
            🟢 Green = text was found, 🔴 Red = text was not found.
            The y-axis shows the iteration number, and the x-axis shows the timeline.
            Watch how the search narrows down to find the first appearance!
            """)

        # Binary search log (collapsed)
        with st.expander("🔍 Detailed Binary Search Log"):
            for log_entry in result['search_log']:
                st.text(log_entry)

        # Visual diff of rendered first paragraph
        st.subheader("Visual Diff - First Paragraph")
        with st.spinner("Loading rendered content..."):
            current_html = wiki.get_rendered_html(st.session_state.page_title, rev['revid'])

            # Get parent revision ID first
            parent_id = None
            try:
                params = {'action': 'query', 'revids': rev['revid'], 'prop': 'revisions', 'rvprop': 'ids', 'format': 'json'}
                resp = wiki.session.get(wiki.api_endpoint, params=params)
                data = resp.json()
                pages = data.get('query', {}).get('pages', {})
                if pages:
                    page = list(pages.values())[0]
                    if 'revisions' in page:
                        parent_id = page['revisions'][0].get('parentid')
            except:
                pass

            if parent_id and current_html:
                parent_html = wiki.get_rendered_html(st.session_state.page_title, parent_id)

                if parent_html:
                    # Extract first paragraphs (HTML)
                    old_para_html = wiki.extract_first_paragraph_html(parent_html)
                    new_para_html = wiki.extract_first_paragraph_html(current_html)

                    # Strip HTML tags to get plain text
                    old_text = re.sub(r'<[^>]+>', '', old_para_html)
                    new_text = re.sub(r'<[^>]+>', '', new_para_html)

                    # Simple diff using difflib on plain text
                    old_words = old_text.split()
                    new_words = new_text.split()

                    matcher = difflib.SequenceMatcher(None, old_words, new_words)

                    old_display = []
                    new_display = []

                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == 'equal':
                            old_display.extend(old_words[i1:i2])
                            new_display.extend(new_words[j1:j2])
                        elif tag == 'delete':
                            for word in old_words[i1:i2]:
                                old_display.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{word}</span>')
                        elif tag == 'insert':
                            for word in new_words[j1:j2]:
                                new_display.append(f'<span style="background-color: #ccffcc;">{word}</span>')
                        elif tag == 'replace':
                            for word in old_words[i1:i2]:
                                old_display.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{word}</span>')
                            for word in new_words[j1:j2]:
                                new_display.append(f'<span style="background-color: #ccffcc;">{word}</span>')

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Before Edit**")
                        st.markdown(f'<div style="padding: 15px; border: 1px solid #a2a9b1; background: white; font-family: sans-serif; line-height: 1.6;">{" ".join(old_display)}</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown("**After Edit**")
                        st.markdown(f'<div style="padding: 15px; border: 1px solid #a2a9b1; background: white; font-family: sans-serif; line-height: 1.6;">{" ".join(new_display)}</div>', unsafe_allow_html=True)

                    st.markdown('<p style="margin-top: 10px;"><span style="background-color: #ccffcc; padding: 3px 8px;">🟢 Added</span> <span style="background-color: #ffcccc; padding: 3px 8px; margin-left: 10px;">🔴 Removed</span></p>', unsafe_allow_html=True)
                else:
                    st.warning("Could not load previous revision for visual diff")
            else:
                st.warning("Could not load rendered content for visual diff")

        # Wikitext diff section
        st.subheader("Wikitext Diff")

        # Toggle for inline vs side-by-side
        diff_mode = st.radio("Display mode:", ["Side-by-side", "Inline"], horizontal=True, key="diff_mode")

        with st.spinner("Loading diff..."):
            diff_html = wiki.get_revision_diff_html(st.session_state.page_title, rev['revid'], inline=(diff_mode == "Inline"))
            if diff_html:
                # Add Wikipedia diff CSS
                wiki_diff_css = """
                <style>
                .diff {
                    border: none;
                    width: 100%;
                    border-collapse: collapse;
                    font-family: monospace;
                    font-size: 13px;
                }
                .diff td {
                    padding: 2px 5px;
                }
                .diff-deletedline, .diff-addedline {
                    vertical-align: top;
                }
                .diff-side-deleted {
                    background-color: #ffe49c;
                }
                .diff-side-added {
                    background-color: #ffe49c;
                }
                .diff-deletedline {
                    background-color: #fff5e6;
                }
                .diff-addedline {
                    background-color: #fff5e6;
                }
                .diff-context {
                    background-color: #f9f9f9;
                    color: #333;
                }
                .diff-marker {
                    text-align: right;
                    font-weight: bold;
                    padding: 2px 5px;
                }
                del.diffchange {
                    background-color: #feeec8;
                    text-decoration: line-through;
                }
                ins.diffchange {
                    background-color: #d8ecff;
                    text-decoration: none;
                }
                .diff-lineno {
                    font-weight: bold;
                    background-color: #eaecf0;
                    padding: 2px 10px;
                }
                /* Inline diff styles */
                .mw-diff-inline-header {
                    font-weight: bold;
                    background-color: #eaecf0;
                    padding: 5px;
                    margin-top: 10px;
                }
                .mw-diff-inline-context {
                    background-color: #f9f9f9;
                    padding: 2px 5px;
                }
                .mw-diff-inline-changed {
                    background-color: #fff5e6;
                    padding: 2px 5px;
                }
                </style>
                """
                st.markdown(wiki_diff_css, unsafe_allow_html=True)

                if diff_mode == "Inline":
                    st.markdown(f'<div>{diff_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<table class="diff">{diff_html}</table>', unsafe_allow_html=True)
            else:
                st.error("Could not load diff visualization")

        # Disclaimer
        st.warning("""
        ⚠️ **Important Note:** This tool finds the *first appearance* within the searched time range.
        The text may have:
        - Been added before the search range
        - Been removed and then re-added
        - Been modified after this edit

        For complete history, check the full revision history on Wikipedia.
        """)


if __name__ == "__main__":
    main()
