import streamlit as st
import plotly.express as px
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import emoji
from collections import Counter
# --- ADD THESE IMPORTS ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (this fixes the "resource not found" error)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


# --- Page Configuration ---
st.set_page_config(page_title="SOMA - Social Media Analyzer", layout="wide")

st.title("Social Media Analyzer (SOMA)")
st.markdown("### WhatsApp Group Chat Analysis System")

# --- Preprocessing Functions (Modified for Streamlit) ---

# --- IMPROVED Preprocessing Function ---

# --- IMPROVED Preprocessing Function ---

# --- IMPROVED Preprocessing Function ---

# --- IMPROVED Preprocessing Function ---

def process_chat_data(uploaded_file):
    """
    Universal parser for both Android and iPhone formats (12-hour & 24-hour).
    """
    # 1. Read and Decode
    stringio = uploaded_file.getvalue().decode("utf-8-sig")
    data = stringio.splitlines()

    # 2. Define Universal Patterns
    patterns = [
        # Pattern A: iPhone (Brackets)
        re.compile(r'^\[(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)(?:\s?([a-zA-Z]{2}))?\]\s(.*?):\s?(.*)'),

        # Pattern B: Android (No Brackets)
        re.compile(r'^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)(?:\s?([a-zA-Z]{2}))?\s-\s(.*?):\s?(.*)')
    ]

    # 3. Detect which pattern works
    chosen_pattern = None
    for line in data[:50]:
        # CLEAN: Remove all invisible BiDi characters and narrow spaces that break iOS parsing
        clean_line = re.sub(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069\u202f]', '', line).strip()
        for p in patterns:
            if p.match(clean_line):
                chosen_pattern = p
                break
        if chosen_pattern:
            break
            
    if not chosen_pattern:
        st.error("Could not detect a valid date format. Please check your file.")
        return None

    # 4. Parse the file
    messages = []
    current_message = None

    for line in data:
        # Strip invisible formatting characters dynamically
        clean_line = re.sub(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069\u202f]', '', line).strip()
        
        match = chosen_pattern.match(clean_line)
        if match:
            # Save previous message
            if current_message:
                messages.append(current_message)
            
            date, time, ampm, sender, message = match.groups()
            
            # Format time explicitly to help pandas calculate 12h/24h correctly
            clean_time = time.strip()
            if ampm:  
                clean_time = f"{clean_time} {ampm.strip().upper()}"
                
            current_message = [date, clean_time, sender, message]
        else:
            # Append multiline messages
            if current_message and clean_line:
                current_message[3] += f" {clean_line}"

    if current_message:
        messages.append(current_message)

    # 5. Create DataFrame
    df = pd.DataFrame(messages, columns=['Date', 'Time', 'Sender', 'Message'])
    
    if df.empty:
        return None

    # ---------------------------------------------------------
    # NEW FIX: EXPANDED SYSTEM FILTERS FOR IPHONE MEDIA
    # ---------------------------------------------------------
    system_filters = [
        r"Media omitted",
        r"sticker omitted",
        r"image omitted",
        r"video omitted",
        r"video note omitted",
        r"audio omitted",
        r"contact card omitted",
        r"document omitted",
        r"Messages and calls are end-to-end encrypted",
        r"created group",
        r"added you",
        r"added ",
        r"changed the subject",
        r"You're now an admin"
    ]
    
    # case=False ensures it catches "Contact card omitted" or "contact card omitted" equally
    df = df[~df['Message'].str.contains('|'.join(system_filters), regex=True, case=False, na=False)]
    # ---------------------------------------------------------

    # 6. DateTime Conversion
    try:
        df['DateTimeStr'] = df['Date'] + ' ' + df['Time']
        
        # SMART DATE PARSING: Check if the first number in the date is > 12
        first_date_num = df['Date'].str.extract(r'^(\d{1,2})')[0].astype(float)
        if first_date_num.max() > 12:
            # The first number goes above 12, so the format MUST be DD/MM/YYYY
            df['DateTime'] = pd.to_datetime(df['DateTimeStr'], dayfirst=True, errors='coerce')
        else:
            # The first number maxes out at 12, so the format MUST be MM/DD/YYYY
            df['DateTime'] = pd.to_datetime(df['DateTimeStr'], dayfirst=False, errors='coerce')
        
        # Drop unparseable rows safely
        df = df.dropna(subset=['DateTime'])
        
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.day_name()
        df['Message_Length'] = df['Message'].apply(lambda x: len(str(x).split()))
        
        # --- SENTIMENT ANALYSIS ---
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        df['Sentiment_Score'] = df['Message'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
        df['Sentiment_Category'] = df['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
        # --------------------------

    except Exception as e:
        st.error(f"Error converting dates: {e}")
    
    return df

# --- Helper Analysis Functions ---

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

def get_most_common_emoji(text_series):
    all_emojis = []
    for text in text_series:
        all_emojis.extend(extract_emojis(text))
    if not all_emojis:
        return "None"
    return Counter(all_emojis).most_common(1)[0][0]

# --- Main App Flow ---

# 1. User Prompt & Upload
uploaded_file = st.file_uploader("Upload your WhatsApp Chat (.txt file)", type="txt")

if uploaded_file is not None:
    # Process data
    df = process_chat_data(uploaded_file)
    
    if df is not None and not df.empty:
        st.success("File uploaded and processed successfully!")
        
# --- Chat Summary Section (Added Request) ---
        st.markdown("### 📊 Chat Overview")
        
        # Calculate statistics
        total_messages = len(df)
        first_date = df['DateTime'].min()
        last_date = df['DateTime'].max()
        duration = (last_date - first_date).days
        
        # Create 3 columns for a dashboard look
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        with col_sum1:
            st.metric(label="Total Messages", value=f"{total_messages:,}")
            
        with col_sum2:
            st.metric(label="Total Days", value=f"{duration} days")
            
        with col_sum3:
            # Format: Day/Month/Year
            date_range = f"{first_date.strftime('%d/%m/%y')} - {last_date.strftime('%d/%m/%y')}"
            st.metric(label="Timeframe", value=date_range)

        # --- Navigation (Three Choices) ---
        st.markdown("---")
        st.subheader("Select Analysis Module")
        
        # Using columns for button-like layout or tabs for cleaner navigation
        tab1, tab2, tab3 , tab4= st.tabs(["📊 Volume Analysis", "📝 Content Analysis", "⏰ Temporal Analysis", "Sentiment Analysis"])

        # === CHOICE 1: VOLUME ANALYSIS ===
        with tab1:
            st.header("Volume Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Message Count by User")
                user_counts = df['Sender'].value_counts().head(10)
                fig, ax = plt.subplots()
                sns.barplot(x=user_counts.values, y=user_counts.index, ax=ax, palette="viridis")
                ax.set_xlabel("Number of Messages")
                st.pyplot(fig)

            with col2:
                st.subheader("Message Distribution")
                fig2, ax2 = plt.subplots()
                ax2.pie(user_counts.values, labels=user_counts.index, autopct='%1.1f%%', startangle=90)
                ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig2)
            
            st.subheader("Daily Message Trend")
            daily_counts = df.groupby(df['DateTime'].dt.date).size()
            st.line_chart(daily_counts)

        # === CHOICE 2: CONTENT ANALYSIS ===
        # === CHOICE 2: CONTENT ANALYSIS ===
        # === CHOICE 2: CONTENT ANALYSIS ===
        with tab2:
            st.header("Content Analysis")
            
            # 1. Word Cloud
            st.subheader("Most Common Words (Word Cloud)")
            text_data = " ".join(msg for msg in df['Message'])
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text_data)
            
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            # 2. Top 5 Words Analysis Table
            st.markdown("---")
            st.subheader("Top 5 Common Words & Who Used Them")
            
            # Tokenize and clean text
            all_text_lower = " ".join(df['Message'].astype(str)).lower()
            words = re.findall(r'\b\w+\b', all_text_lower)
            
            # Remove stopwords and short words
            filtered_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
            
            if filtered_words:
                top_5_words = Counter(filtered_words).most_common(5)
                
                word_usage_list = []
                for word, total_count in top_5_words:
                    # Find which users used this word
                    mask = df['Message'].str.contains(r'\b' + re.escape(word) + r'\b', case=False, regex=True, na=False)
                    user_counts = df[mask]['Sender'].value_counts()
                    
                    top_users_str = ", ".join([f"{u} ({c})" for u, c in user_counts.head(3).items()])
                    
                    word_usage_list.append({
                        "Word": word,
                        "Total Frequency": total_count,
                        "Top Users (Count)": top_users_str
                    })
                
                df_top_words = pd.DataFrame(word_usage_list)
                
                # --- FIX: Start Index at 1 ---
                df_top_words.index = df_top_words.index + 1
                
                st.table(df_top_words)
            else:
                st.info("Not enough text data to analyze common words.")
            
            # 3. User Contribution Details
            st.markdown("---")
            st.subheader("User Contribution Details")
            
            user_stats = []
            grouped = df.groupby('Sender')
            
            for sender, group in grouped:
                total_words = group['Message_Length'].sum()
                avg_words = group['Message_Length'].mean()
                longest_msg_len = group['Message_Length'].max()
                most_used_emoji = get_most_common_emoji(group['Message'])
                
                user_stats.append({
                    "User": sender,
                    "Total Words": total_words,
                    "Avg Words/Msg": round(avg_words, 1),
                    "Longest Msg (Words)": longest_msg_len,
                    "Top Emoji": most_used_emoji
                })
            
            stats_df = pd.DataFrame(user_stats).sort_values(by="Total Words", ascending=False)
            
            # --- FIX: Reset Index and Start at 1 ---
            stats_df.reset_index(drop=True, inplace=True) # Reset index after sorting
            stats_df.index = stats_df.index + 1           # Add 1 to start from 1
            
            st.dataframe(stats_df, use_container_width=True)

        # === CHOICE 3: TEMPORAL ANALYSIS ===
        with tab3:
            st.header("Temporal Analysis")
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.subheader("Activity by Hour of Day")
                hourly_counts = df['Hour'].value_counts().sort_index()
                st.bar_chart(hourly_counts)
                
                # Identify Peak Hour
                peak_hour = hourly_counts.idxmax()
                st.metric(label="Peak Activity Hour", value=f"{peak_hour}:00")

            with col_t2:
                st.subheader("Activity by Day of Week")
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = df['DayOfWeek'].value_counts().reindex(day_order)
                fig_day, ax_day = plt.subplots()
                sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax_day, palette="Blues_d")
                plt.xticks(rotation=45)
                st.pyplot(fig_day)

            # Heatmap (Hour vs Day)
            st.subheader("Activity Heatmap (Hour vs. Day)")
            heatmap_data = df.pivot_table(index='DayOfWeek', columns='Hour', aggfunc='size', fill_value=0)
            
            # Reorder index to ensure Monday comes first
            heatmap_data = heatmap_data.reindex(day_order)
            
            fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
            
            # --- CHANGE IS HERE: annot=False removes the numbers ---
            sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax_heat, annot=False, fmt='g')
            
            st.pyplot(fig_heat)

            # --- NEW: Add the explanation box right below the heatmap ---
            st.info("💡 **How to read this chart:** The darker boxes represent times when the group chat is highly active, while the lighter boxes represent periods when the chat is less active.")


            # --- PASTE THIS NEW LINE GRAPH CODE HERE ---
            st.markdown("---")
            st.subheader("Hourly Activity Trajectory by Day")
            
            # Transpose (.T) the heatmap_data so Hours become the X-axis 
            # and Days become the different colored lines
            line_chart_data = heatmap_data.T
            
            # Use Streamlit's native line chart for interactive hovering
            st.line_chart(line_chart_data)
            # ---------------------------------------------
            # --- PASTE THIS NEW PEAK FINDER CODE HERE ---
            st.markdown("---")
            st.subheader("🔥 Peak Activity Insight")
            
            # Find the exact day and hour with the maximum messages
            # .stack() turns the 2D grid into a 1D list, and idxmax() finds the index of the highest value
            peak_index = heatmap_data.stack().idxmax()
            peak_day = peak_index[0]
            peak_hour = peak_index[1]
            peak_messages = heatmap_data.max().max()
            
            # Format the time nicely (e.g., making 14 into 14:00)
            formatted_time = f"{peak_hour:02d}:00"
            
            # Display the insight in a highlighted success box
            st.success(f"**Busiest Time:** The group chat reaches its absolute peak on **{peak_day}s at {formatted_time}**, with a total of **{int(peak_messages)} messages** sent during this hour.")
            # ---------------------------------------------

            # === CHOICE 4: SENTIMENT ANALYSIS (NEW) ===
        with tab4:
            st.header("Sentiment Analysis Overview")
            
            # 1. Overall Sentiment Distribution
            st.subheader("What is the general mood of the group?")
            sentiment_counts = df['Sentiment_Category'].value_counts()
            
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                fig_sent, ax_sent = plt.subplots()
                # Custom colors: Green for pos, Grey for neu, Red for neg
                colors = {'Positive': '#66b3ff', 'Neutral': '#99ff99', 'Negative': '#ff9999'}
                
                ax_sent.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, 
                            colors=[colors.get(x, '#999999') for x in sentiment_counts.index])
                ax_sent.axis('equal')
                st.pyplot(fig_sent)
            
            with col_s2:
                st.markdown("**Interpretation:**")
                st.markdown(f"- **Positive:** {sentiment_counts.get('Positive', 0)} messages")
                st.markdown(f"- **Negative:** {sentiment_counts.get('Negative', 0)} messages")
                if sentiment_counts.get('Negative', 0) > sentiment_counts.get('Positive', 0):
                    st.warning("⚠️ This group tends to be more negative/critical.")
                else:
                    st.success("✅ This group is generally positive!")

            # 2. Sentiment Over Time (The "Story" Graph)
            st.markdown("---")
            st.subheader("Mood Swing Over Time")
            
            # Create a dataframe for the daily sentiment
            daily_sentiment_df = df.groupby(df['DateTime'].dt.date)['Sentiment_Score'].mean().reset_index()
            daily_sentiment_df.rename(columns={'DateTime': 'Date', 'Sentiment_Score': 'Average Sentiment'}, inplace=True)

            # Find the exact dates of the highest and lowest peaks
            max_date = daily_sentiment_df.loc[daily_sentiment_df['Average Sentiment'].idxmax()]['Date']
            min_date = daily_sentiment_df.loc[daily_sentiment_df['Average Sentiment'].idxmin()]['Date']

            # Get the "most extreme" message from those specific days
            best_msg = df[df['DateTime'].dt.date == max_date].sort_values(by='Sentiment_Score', ascending=False).iloc[0]['Message']
            worst_msg = df[df['DateTime'].dt.date == min_date].sort_values(by='Sentiment_Score', ascending=True).iloc[0]['Message']

            # Create a function to add the specific message only to the peak days
            def get_hover_text(row):
                if row['Date'] == max_date:
                    return f"🏆 Top Message: '{best_msg}'"
                elif row['Date'] == min_date:
                    return f"💔 Worst Message: '{worst_msg}'"
                else:
                    return "(Hover over the highest or lowest peak to see the message)"

            daily_sentiment_df['Hover_Text'] = daily_sentiment_df.apply(get_hover_text, axis=1)

            # Build the interactive Plotly graph
            fig_sent_line = px.line(
                daily_sentiment_df, 
                x='Date', 
                y='Average Sentiment', 
                markers=True,
                custom_data=['Hover_Text']
            )

            # Format the tooltip to look clean and professional
            fig_sent_line.update_traces(
                hovertemplate="<b>Date:</b> %{x}<br><b>Avg Score:</b> %{y:.2f}<br><br><b>%{customdata[0]}</b><extra></extra>",
                line=dict(color='#66b3ff', width=3),
                marker=dict(size=8)
            )

            # Display in Streamlit
            st.plotly_chart(fig_sent_line, use_container_width=True)
            # --- PASTE THIS EXPLANATION BOX CODE HERE ---
            st.info(
            "💡 **Understanding Sentiment Scores:**\n\n"
            "The system uses Natural Language Processing (NLP) to read the text and emojis in each message and assigns an emotional score:\n"
            "* 🟢 **+1.0 to > 0.0:** Positive sentiment (happy, supportive, or excited messages)\n"
            "* ⚪ **0.0:** Neutral sentiment (informational, short, or unemotional messages)\n"
            "* 🔴 **< 0.0 to -1.0:** Negative sentiment (angry, sad, or frustrated messages)\n\n"
            "The messages shown on the graph represent the text that heavily influenced the average score for that specific day."
            )
            # ---------------------------------------------
            st.caption("A score > 0 is Positive, < 0 is Negative. Hover over the highest and lowest points to read the exact message that caused the spike/drop!")

            # 3. Who is the most Positive/Negative person?
            st.markdown("---")
            st.subheader("User Positivity Ranking")
            
            user_sentiment = df.groupby('Sender')['Sentiment_Score'].mean().sort_values(ascending=False)
            
            st.bar_chart(user_sentiment)
            st.write(f"🏆 **Most Positive User:** {user_sentiment.idxmax()}")
            st.write(f"🌩️ **Most Critical User:** {user_sentiment.idxmin()}")
        
        # [EXISTING CODE ABOVE ENDS HERE]
            
        # --- PASTE THIS NEW SECTION HERE ---
        # --- SIDEBAR: EXPORT & REPORT GENERATION ---
        with st.sidebar:
            st.markdown("---")
            st.header("📂 Export & Save")
            
            # 1. GENERATE TEXT REPORT FUNCTION
            def generate_insight_report(dataframe):
                # Basic Stats
                total_msg = len(dataframe)
                users = dataframe['Sender'].nunique()
                start_date = dataframe['DateTime'].min().strftime('%d/%m/%Y')
                end_date = dataframe['DateTime'].max().strftime('%d/%m/%Y')
                
                # Activity Stats
                busiest_day = dataframe['DayOfWeek'].value_counts().idxmax()
                peak_hour = dataframe['Hour'].value_counts().idxmax()
                
                # Top Users
                top_users = dataframe['Sender'].value_counts().head(3).to_dict()
                top_users_str = "\n".join([f"   - {user}: {count} msgs" for user, count in top_users.items()])
                
                # Sentiment Stats (if available)
                sentiment_summary = "Not calculated"
                if 'Sentiment_Category' in dataframe.columns:
                    pos = len(dataframe[dataframe['Sentiment_Category'] == 'Positive'])
                    neg = len(dataframe[dataframe['Sentiment_Category'] == 'Negative'])
                    sentiment_summary = f"Positive: {pos} | Negative: {neg}"

                # Construct the Report Text
                report = f"""
=============================================
   SOMA - SOCIAL MEDIA ANALYZER REPORT
=============================================
Generated on: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}

--- 1. OVERVIEW ---
Total Messages: {total_msg}
Active Users: {users}
Date Range: {start_date} to {end_date}

--- 2. ACTIVITY INSIGHTS ---
Busiest Day: {busiest_day}
Peak Hour: {peak_hour}:00
(This suggests the group is most active on {busiest_day}s around {peak_hour}:00)

--- 3. TOP CONTRIBUTORS ---
{top_users_str}

--- 4. SENTIMENT SUMMARY ---
{sentiment_summary}
(Higher positive numbers indicate a generally happy group)

=============================================
End of Report
"""
                return report

            # 2. CREATE DOWNLOAD BUTTONS
            
            # A. Download Text Report
            report_content = generate_insight_report(df)
            st.download_button(
                label="📄 Download Analysis Report (.txt)",
                data=report_content,
                file_name="SOMA_Analysis_Report.txt",
                mime="text/plain"
            )
            
            # B. Download Cleaned Data (CSV)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Processed Data (.csv)",
                data=csv_data,
                file_name="processed_chat_data.csv",
                mime="text/csv",
                help="Download the cleaned data to open in Excel."
            )
            
            # C. Instructions for Printing Charts
            st.info("🖨️ **To Print Charts:**\nPress `Ctrl + P` (or Cmd + P) in your browser and select **'Save as PDF'** to save the graphs.")

    else:
        st.error("Could not parse the file. Please ensure it is a valid WhatsApp text export.")