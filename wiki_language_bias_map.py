"""
Wikipedia Language Bias Map
Compare how different language versions of the same Wikipedia article portray Israel
"""

import streamlit as st
import pandas as pd
import requests
import time
import anthropic
import os
import plotly.graph_objects as go
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


# Language to country/region mapping
LANGUAGE_TO_COUNTRY = {
    'en': 'United States',  # English (using US as primary)
    'ar': 'Saudi Arabia',  # Arabic
    'he': 'Israel',  # Hebrew
    'fr': 'France',
    'de': 'Germany',
    'es': 'Spain',
    'ru': 'Russia',
    'ja': 'Japan',
    'zh': 'China',
    'pt': 'Brazil',
    'it': 'Italy',
    'nl': 'Netherlands',
    'pl': 'Poland',
    'tr': 'Turkey',
    'sv': 'Sweden',
    'fa': 'Iran',  # Persian
    'ko': 'South Korea',
    'id': 'Indonesia',
    'cs': 'Czech Republic',
    'fi': 'Finland',
    'hu': 'Hungary',
    'no': 'Norway',
    'da': 'Denmark',
    'ro': 'Romania',
    'uk': 'Ukraine',
    'el': 'Greece',
    'bg': 'Bulgaria',
    'sr': 'Serbia',
    'hr': 'Croatia',
    'sk': 'Slovakia',
    'lt': 'Lithuania',
    'sl': 'Slovenia',
    'et': 'Estonia',
    'lv': 'Latvia',
    'th': 'Thailand',
    'vi': 'Vietnam',
    'ms': 'Malaysia',
    'hi': 'India',
    'bn': 'Bangladesh',
    'ur': 'Pakistan',
    'ta': 'India',
    'te': 'India',
    'ml': 'India',
    'ka': 'Georgia',
    'az': 'Azerbaijan',
    'hy': 'Armenia',
    'be': 'Belarus',
    'af': 'South Africa',
    'sq': 'Albania',
    'eu': 'Spain',  # Basque
    'ca': 'Spain',  # Catalan
    'cy': 'United Kingdom',  # Welsh
    'ga': 'Ireland',
    'is': 'Iceland',
    'mk': 'North Macedonia',
    'mt': 'Malta',
    'sw': 'Kenya',
}


class WikiLanguageBiasAnalyzer:
    """Analyzes bias across Wikipedia language versions"""

    def __init__(self, api_key=None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikiLanguageBiasMap/1.0 (Educational Research)'
        })
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY_NC"))

    def get_language_versions(self, page_title, source_lang='en'):
        """
        Get all language versions of a Wikipedia article

        Args:
            page_title: Article title (e.g., "October 7 attacks")
            source_lang: Source language code (default: 'en')

        Returns:
            dict: {lang_code: page_title}
        """
        api_url = f"https://{source_lang}.wikipedia.org/w/api.php"

        params = {
            'action': 'query',
            'titles': page_title,
            'prop': 'langlinks',
            'lllimit': 500,
            'format': 'json'
        }

        try:
            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data['query']['pages']
            page_id = list(pages.keys())[0]

            if page_id == '-1':
                return {}

            langlinks = pages[page_id].get('langlinks', [])

            # Convert to dict
            result = {source_lang: page_title}  # Include source language
            for link in langlinks:
                result[link['lang']] = link['*']

            return result

        except Exception as e:
            st.error(f"Error fetching language versions: {e}")
            return {}

    def get_first_paragraph(self, page_title, lang_code):
        """
        Get the first paragraph of a Wikipedia article

        Args:
            page_title: Article title
            lang_code: Language code (e.g., 'en', 'fr', 'ar')

        Returns:
            str: First paragraph text
        """
        api_url = f"https://{lang_code}.wikipedia.org/w/api.php"

        params = {
            'action': 'query',
            'titles': page_title,
            'prop': 'extracts',
            'exintro': True,  # Only intro section
            'explaintext': True,  # Plain text
            'format': 'json'
        }

        try:
            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data['query']['pages']
            page_id = list(pages.keys())[0]

            if page_id == '-1':
                return None

            extract = pages[page_id].get('extract', '')

            # Get first paragraph (split by double newline)
            paragraphs = extract.split('\n\n')
            first_para = paragraphs[0] if paragraphs else extract

            return first_para.strip()

        except Exception as e:
            return None

    def analyze_batch_with_llm(self, articles_data):
        """
        Analyze multiple language versions in one LLM call for consistency

        Args:
            articles_data: List of dicts with keys: lang_code, lang_name, text

        Returns:
            dict: {lang_code: {'bias_score': float, 'explanation': str}}
        """
        # Build prompt with all articles
        prompt = """Analyze these Wikipedia article introductions for anti-Israel bias.

Rate each on a scale of 1-10, where:
- **1-2**: Pro-Israel or balanced
- **3-4**: Slightly anti-Israel
- **5-6**: Moderately anti-Israel
- **7-8**: Strongly anti-Israel
- **9-10**: Extremely anti-Israel

Consider:
1. **Language/framing**: "genocide" vs "war", "occupation" vs "disputed territories"
2. **One-sidedness**: Only Palestinian casualties? Israeli victims ignored?
3. **Context**: Hamas terrorism mentioned? October 7th?
4. **Loaded terms**: "Zionist entity", "settler-colonial", inflammatory language
5. **Facts vs opinion**: Presenting contested claims as facts

---

"""
        for i, article in enumerate(articles_data[:20], 1):  # Limit to 20 per batch
            prompt += f"""**Article {i}: {article['lang_name']} ({article['lang_code']})**
{article['text'][:1000]}

---

"""

        prompt += """
Respond in this EXACT format for each article:
ARTICLE_1
Score: [number 1-10]
Explanation: [2-3 sentences]

ARTICLE_2
Score: [number 1-10]
Explanation: [2-3 sentences]

etc."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            # DEBUG: Store raw response for debugging
            if hasattr(st, 'session_state'):
                st.session_state.last_llm_response = response_text

            # Parse results
            results = {}
            import re

            # Split by ARTICLE_ markers
            article_blocks = re.split(r'ARTICLE[_\s]*(\d+)', response_text)

            # Process blocks (every 2 items: number, content)
            for i in range(1, len(article_blocks), 2):
                if i + 1 >= len(article_blocks):
                    break

                article_num = int(article_blocks[i])
                article_content = article_blocks[i + 1]

                # Get corresponding article data
                if 1 <= article_num <= len(articles_data):
                    article = articles_data[article_num - 1]

                    # Extract score and explanation
                    score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', article_content)
                    explanation_match = re.search(r'Explanation:\s*(.+?)(?=\n\n|ARTICLE_|$)', article_content, re.DOTALL)

                    if score_match and explanation_match:
                        score = float(score_match.group(1))
                        explanation = explanation_match.group(1).strip()
                    else:
                        # Try alternative parsing
                        lines = article_content.strip().split('\n')
                        score = 5.0
                        explanation = "Could not parse response"

                        for line in lines:
                            if 'score' in line.lower() and re.search(r'\d+', line):
                                score_num = re.search(r'(\d+(?:\.\d+)?)', line)
                                if score_num:
                                    score = float(score_num.group(1))
                            elif 'explanation' in line.lower():
                                explanation = line.split(':', 1)[1].strip() if ':' in line else line.strip()

                    results[article['lang_code']] = {
                        'bias_score': min(10, max(1, score)),
                        'explanation': explanation
                    }

            # Fill in any missing results
            for article in articles_data:
                if article['lang_code'] not in results:
                    results[article['lang_code']] = {
                        'bias_score': 5.0,
                        'explanation': 'Response parsing failed - article not found in LLM response'
                    }

            return results

        except Exception as e:
            # Fallback: return neutral scores
            return {
                article['lang_code']: {
                    'bias_score': 5.0,
                    'explanation': f'Error analyzing: {str(e)}'
                }
                for article in articles_data
            }

    def translate_titles_batch(self, titles_dict):
        """
        Translate multiple titles to English in one LLM call

        Args:
            titles_dict: {lang_code: title_text}

        Returns:
            dict: {lang_code: translated_title}
        """
        if not titles_dict:
            return {}

        # Build prompt with all titles
        prompt = "Translate these Wikipedia article titles to English. Provide literal translations.\n\n"

        for lang_code, title in titles_dict.items():
            prompt += f"{lang_code}: {title}\n"

        prompt += "\nRespond in this exact format:\nen: [English translation]\nfr: [English translation]\n..."

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            # Parse results
            import re
            results = {}
            for lang_code in titles_dict.keys():
                pattern = rf"{lang_code}:\s*(.+?)(?:\n|$)"
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    results[lang_code] = match.group(1).strip()
                else:
                    results[lang_code] = titles_dict[lang_code]  # Fallback to original

            return results

        except Exception as e:
            # Fallback: return original titles
            return {lang: title for lang, title in titles_dict.items()}

    def translate_text(self, text, from_lang):
        """
        Translate text to English using LLM

        Args:
            text: Text to translate
            from_lang: Source language code

        Returns:
            str: Translated text
        """
        prompt = f"""Translate this text from {from_lang} to English. Provide only the translation, no explanation.

Text:
{text[:3000]}"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text.strip()

        except Exception as e:
            return f"Translation error: {str(e)}"


def main():
    st.set_page_config(
        page_title="Wikipedia Language Bias Map",
        page_icon="🗺️",
        layout="wide"
    )

    st.title("🗺️ Wikipedia Language Bias Map")
    st.markdown("Compare how different language versions of Wikipedia articles portray Israel")

    # Sidebar
    st.sidebar.header("Configuration")

    article_input = st.sidebar.text_input(
        "Wikipedia Article",
        value="Gaza_genocide",
        help="Enter article name or paste full URL"
    )

    # Extract article name from URL if needed
    if 'wikipedia.org/wiki/' in article_input:
        article_name = article_input.split('/wiki/')[-1]
    else:
        article_name = article_input

    max_languages = st.sidebar.slider(
        "Max Languages to Analyze",
        min_value=0,
        max_value=50,
        value=20,
        step=5,
        help="0 = analyze all available languages. Costs ~$0.20-0.50 per batch of 20."
    )

    # Initialize
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = WikiLanguageBiasAnalyzer()
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Fetch button
    if st.sidebar.button("🌍 Analyze Language Versions", type="primary"):
        with st.spinner("Fetching language versions..."):
            lang_versions = st.session_state.analyzer.get_language_versions(article_name)

            if not lang_versions:
                st.error("Article not found or has no language versions")
            else:
                st.success(f"Found {len(lang_versions)} language versions!")

                # Limit to max_languages (0 = all)
                if max_languages > 0 and len(lang_versions) > max_languages:
                    # Prioritize major languages
                    priority_langs = ['en', 'ar', 'he', 'fr', 'de', 'es', 'ru', 'zh', 'ja', 'pt']
                    selected = {}
                    for lang in priority_langs:
                        if lang in lang_versions:
                            selected[lang] = lang_versions[lang]

                    # Fill remaining
                    remaining = max_languages - len(selected)
                    for lang, title in lang_versions.items():
                        if lang not in selected and remaining > 0:
                            selected[lang] = title
                            remaining -= 1

                    lang_versions = selected

                st.info(f"Fetching text for {len(lang_versions)} languages in parallel...")

                # STEP 1: Fetch all first paragraphs IN PARALLEL
                articles_data = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Create analyzer instance for parallel workers (can't use session_state in threads)
                analyzer = st.session_state.analyzer

                def fetch_paragraph(lang_code, page_title, analyzer_instance):
                    """Helper to fetch one paragraph"""
                    para = analyzer_instance.get_first_paragraph(page_title, lang_code)
                    if para and len(para) > 50:
                        return {
                            'lang_code': lang_code,
                            'lang_name': page_title,
                            'text': para
                        }
                    return None

                # Parallel fetching
                completed = 0
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_lang = {
                        executor.submit(fetch_paragraph, lang_code, page_title, analyzer): lang_code
                        for lang_code, page_title in lang_versions.items()
                    }

                    for future in as_completed(future_to_lang):
                        result = future.result()
                        if result:
                            articles_data.append(result)

                        completed += 1
                        progress_bar.progress(completed / len(lang_versions))
                        status_text.text(f"Fetched {completed}/{len(lang_versions)} articles")

                progress_bar.empty()
                status_text.empty()

                st.success(f"✓ Fetched {len(articles_data)} article versions in parallel")

                # STEP 2: Translate ALL titles in ONE batch LLM call
                st.info("Translating titles in batch...")

                titles_to_translate = {
                    article['lang_code']: article['lang_name']
                    for article in articles_data
                }

                title_translations = st.session_state.analyzer.translate_titles_batch(titles_to_translate)

                st.success("✓ All titles translated in one batch")

                # STEP 3: Batch analyze all articles
                st.info(f"Analyzing {len(articles_data)} articles with LLM...")

                # Process in batches of 20
                all_results = {}
                for i in range(0, len(articles_data), 20):
                    batch = articles_data[i:i+20]
                    batch_results = st.session_state.analyzer.analyze_batch_with_llm(batch)
                    all_results.update(batch_results)
                    time.sleep(0.5)

                st.success("✓ Analysis complete!")

                # STEP 4: Compile results
                results = []
                for article in articles_data:
                    lang_code = article['lang_code']
                    analysis = all_results.get(lang_code, {'bias_score': 5.0, 'explanation': 'Not analyzed'})
                    country = LANGUAGE_TO_COUNTRY.get(lang_code, 'Unknown')

                    results.append({
                        'language_code': lang_code,
                        'original_title': article['lang_name'],
                        'english_title': title_translations.get(lang_code, article['lang_name']),
                        'country': country,
                        'bias_score': analysis['bias_score'],
                        'explanation': analysis['explanation'],
                        'full_text': article['text']
                    })

                st.session_state.results = pd.DataFrame(results)
                st.success(f"✓ Analysis complete! Analyzed {len(results)} language versions.")

    # Display results
    if st.session_state.results is not None:
        df = st.session_state.results

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Languages Analyzed", len(df))
        with col2:
            avg_bias = df['bias_score'].mean()
            st.metric("Average Bias Score", f"{avg_bias:.1f}/10")
        with col3:
            most_biased = df.loc[df['bias_score'].idxmax()]
            st.metric("Most Anti-Israel", f"{most_biased['language_code']} ({most_biased['bias_score']:.1f})")
        with col4:
            least_biased = df.loc[df['bias_score'].idxmin()]
            st.metric("Most Pro-Israel", f"{least_biased['language_code']} ({least_biased['bias_score']:.1f})")

        # Initialize translation cache in session state
        if 'translations' not in st.session_state:
            st.session_state.translations = {}

        # Debug: show raw LLM response if available
        with st.expander("🔍 Debug: View Raw LLM Response"):
            if 'last_llm_response' in st.session_state:
                st.code(st.session_state.last_llm_response, language="text")
            else:
                st.info("No LLM response available yet")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ World Map", "📊 Rankings", "📋 Titles Table", "📝 Full Analysis"])

        with tab1:
            st.header("Bias by Country")

            # Create choropleth map
            fig = go.Figure(data=go.Choropleth(
                locations=df['country'],
                z=df['bias_score'],
                locationmode='country names',
                colorscale=[
                    [0, 'green'],      # Pro-Israel
                    [0.3, 'yellow'],   # Neutral
                    [0.7, 'orange'],   # Anti-Israel
                    [1, 'red']         # Extremely anti-Israel
                ],
                reversescale=False,
                marker_line_color='white',
                marker_line_width=0.5,
                colorbar_title="Bias Score<br>(1=Pro, 10=Anti)",
                zmin=1,
                zmax=10,
                hovertemplate='<b>%{location}</b><br>' +
                             'Language: %{customdata[0]}<br>' +
                             'Bias Score: %{z:.1f}/10<br>' +
                             '<extra></extra>',
                customdata=df[['language_code']].values
            ))

            fig.update_layout(
                title_text=f'Anti-Israel Bias by Language Version: {article_name}',
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth'
                ),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info("ℹ️ **Note:** Countries are mapped based on the primary language of the Wikipedia version analyzed. Each language version is scored independently based on its content.")

        with tab2:
            st.header("Bias Rankings")

            # Sort by bias score
            df_sorted = df.sort_values('bias_score', ascending=False)

            for idx, row in df_sorted.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 3])

                    with col1:
                        if row['bias_score'] >= 7:
                            st.markdown(f"🔴 **{row['bias_score']:.1f}/10**")
                        elif row['bias_score'] >= 5:
                            st.markdown(f"🟡 **{row['bias_score']:.1f}/10**")
                        else:
                            st.markdown(f"🟢 **{row['bias_score']:.1f}/10**")

                    with col2:
                        st.markdown(f"**{row['language_code'].upper()}** - {row['country']}")
                        st.markdown(f"*Original:* {row['original_title']}")
                        st.markdown(f"*English:* {row['english_title']}")

                    with col3:
                        st.write(row['explanation'])

                    st.divider()

        with tab3:
            st.header("Article Titles Translation Table")

            # Create display dataframe
            display_df = df[['language_code', 'country', 'original_title', 'english_title', 'bias_score']].copy()
            display_df = display_df.sort_values('bias_score', ascending=False)
            display_df.columns = ['Lang', 'Country', 'Original Title', 'English Translation', 'Bias Score']

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")
            st.subheader("Translate Full Text")

            # Dropdown to select language
            selected_lang = st.selectbox(
                "Select language to translate full intro paragraph:",
                options=df['language_code'].tolist(),
                format_func=lambda x: f"{x.upper()} - {df[df['language_code']==x]['original_title'].iloc[0]}"
            )

            if selected_lang:
                selected_row = df[df['language_code'] == selected_lang].iloc[0]

                col1, col2 = st.columns([1, 3])

                with col1:
                    if st.button(f"🌐 Translate {selected_lang.upper()} to English"):
                        with st.spinner("Translating..."):
                            translation_key = f"{selected_lang}_full"

                            if translation_key not in st.session_state.translations:
                                translated = st.session_state.analyzer.translate_text(
                                    selected_row['full_text'], selected_lang
                                )
                                st.session_state.translations[translation_key] = translated

                with col2:
                    st.markdown(f"**Original ({selected_lang.upper()}):**")
                    st.write(selected_row['full_text'][:500])

                # Show translation if available
                translation_key = f"{selected_lang}_full"
                if translation_key in st.session_state.translations:
                    st.markdown("---")
                    st.markdown(f"**English Translation:**")
                    st.success(st.session_state.translations[translation_key])

        with tab4:
            st.header("Full Analysis with Text")

            for idx, row in df.sort_values('bias_score', ascending=False).iterrows():
                with st.expander(f"{row['language_code'].upper()} - {row['country']} (Score: {row['bias_score']:.1f})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Original Title:** {row['original_title']}")
                        st.markdown(f"**English Title:** {row['english_title']}")
                        st.markdown(f"**Bias Score:** {row['bias_score']:.1f}/10")

                    with col2:
                        st.markdown(f"**Country:** {row['country']}")
                        st.markdown(f"**Language Code:** {row['language_code']}")

                    st.markdown("---")
                    st.markdown(f"**Analysis:** {row['explanation']}")
                    st.markdown("---")
                    st.markdown("**First Paragraph (Original Language):**")
                    st.write(row['full_text'][:800])


if __name__ == "__main__":
    main()
