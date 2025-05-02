import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # For interactive plots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- Load Data and Train Model with Caching ---
@st.cache_data
def load_data():
    filepath = os.path.join(os.path.dirname(__file__), "IMDB_data.joblib")
    return joblib.load(filepath)

@st.cache_resource
def train_model(X_train, y_train, n_estimators, random_state):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("App Options")

    df = load_data()

    # --- Sidebar Filters ---
    st.sidebar.subheader("Filter Data")

    # Year Range Filter
    min_year = int(df['Release Year'].min())
    max_year = int(df['Release Year'].max())
    year_range = st.sidebar.slider("Release Year", min_year, max_year, (min_year, max_year))
    start_year, end_year = year_range
    filtered_df = df[(df['Release Year'] >= start_year) & (df['Release Year'] <= end_year)]

    # Genre Filter
    all_genres = df['Genre'].str.split(', ').explode().unique()
    selected_genres = st.sidebar.multiselect("Genres", all_genres)
    if selected_genres:
        filtered_df = filtered_df[filtered_df['Genre'].apply(lambda x: any(genre in x for genre in selected_genres))]

    # Rating Filter
    min_rating = float(df['Rating'].min())
    max_rating = float(df['Rating'].max())
    rating_range = st.sidebar.slider("Rating", min_rating, max_rating, (min_rating, max_rating))
    min_r, max_r = rating_range
    filtered_df = filtered_df[(df['Rating'] >= min_r) & (df['Rating'] <= max_r)]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Selection")
    st.sidebar.write(f"Number of Movies: {filtered_df.shape[0]}")
    st.sidebar.write(f"Year Range: {start_year} - {end_year}")
    if selected_genres:
        st.sidebar.write(f"Genres: {', '.join(selected_genres)}")
    st.sidebar.write(f"Rating Range: {min_r:.1f} - {max_r:.1f}")

    tab1, tab2, tab3 = st.tabs(["🏠 Homepage", "📊 Data Analysis", "🎛️ Interactive View"])

    # --- Homepage ---
    with tab1:
        st.title("🎬 Dive into the World of Movies!")
        st.markdown("Ever wondered what makes a movie a hit? Or how ratings have changed over the years? **Unleash your inner film critic and data explorer!** This interactive app invites you to journey through our extensive movie database, uncovering fascinating trends, top-rated gems, and hidden insights. Get ready to explore, visualize, and maybe even discover your next favorite film!")
        st.markdown("---")
        st.subheader("Quick Insights")

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col5 = st.columns(1) # For the movie length distribution

        with col1:
            st.subheader("Top 5 Highest Rated Movies (Filtered)")
            top_5_rated = filtered_df.nlargest(5, 'Rating')[['Title', 'Rating', 'Release Year']]
            st.dataframe(top_5_rated)

            st.subheader("5 Most Recent Movies (Filtered)")
            most_recent_5 = filtered_df.sort_values('Release Year', ascending=False).head(5)[['Title', 'Release Year', 'Rating']]
            st.dataframe(most_recent_5)

        with col2:
            st.subheader("Summary Statistics (Filtered)")
            summary_stats = filtered_df[['Rating', 'Duration', 'Votes']].describe().T
            st.dataframe(summary_stats)

            st.subheader("Top 5 Most Frequent Genres (Filtered)")
            genres_exploded_filtered = filtered_df['Genre'].str.split(', ').explode()
            top_5_genres = genres_exploded_filtered.value_counts().nlargest(5).reset_index()
            top_5_genres.columns = ['Genre', 'Count']
            st.dataframe(top_5_genres)

        with col5[0]: 
            st.subheader("Movie Length Distribution (Filtered)")
            long_movies_filtered = filtered_df[filtered_df['Duration'] > 120]
            short_movies_filtered = filtered_df[filtered_df['Duration'] <= 120]

            long_short_count_filtered = [len(long_movies_filtered), len(short_movies_filtered)]
            categories = ['Long (>120 mins)', 'Short (<=120 mins)']
            colors = sns.color_palette("pastel") # Using a pastel palette for a softer look

            fig_length_pie, ax_length_pie = plt.subplots() # Create figure and axes
            ax_length_pie.pie(long_short_count_filtered, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors)
            ax_length_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title('Proportion of Long and Short Movies (Filtered)')
            st.pyplot(fig_length_pie)

    # --- Data Analysis Tab ---
    with tab2:
        st.title("📊 Data Analysis")

        # Use the filtered_df for analysis
        st.write(f"Number of movies based on current filters: {filtered_df.shape[0]}")

        if not filtered_df.empty:
            # --- Top-Level Metrics ---
            st.subheader("Overview Metrics")
            col_metrics1, col_metrics2, col_metrics3, col_metrics4, col_metrics5 = st.columns(5)
            col_metrics1.metric("Average Rating", f"{filtered_df['Rating'].mean():.2f}")
            col_metrics2.metric("Total Movies", len(filtered_df))
            col_metrics3.metric("Average Duration", f"{filtered_df['Duration'].mean():.2f} min")
            col_metrics4.metric("Average Votes", f"{filtered_df['Votes'].mean():.0f}")
            col_metrics5.metric("Oldest Movie Year", int(filtered_df['Release Year'].min()))
            st.markdown("---")

            col_metrics6 = st.columns(1)
            col_metrics6[0].metric("Newest Movie Year", int(filtered_df['Release Year'].max()))

            with st.expander("Rating Analysis"):
                st.subheader('Rating Distribution')
                fig_rating_dist = plt.figure(figsize=(8, 6))
                sns.histplot(filtered_df['Rating'], bins=20, kde=True, color='skyblue', edgecolor='black')
                plt.title('Distribution of Movie Ratings')
                plt.xlabel('Rating')
                plt.ylabel('Frequency')
                st.pyplot(fig_rating_dist)

                

                st.subheader('Top 10 Movies by Rating')
                top_10_movies = filtered_df.nlargest(10, 'Rating')
                fig_top_10 = plt.figure(figsize=(10, 6))
                sns.barplot(data=top_10_movies, x='Rating', y='Title', palette='Blues_d')
                plt.title('Top 10 Movies by Rating')
                plt.xlabel('Rating')
                plt.ylabel('Movie Title')
                st.pyplot(fig_top_10)

                st.subheader('Bottom 10 Movies by Rating')
                bottom_10_movies = filtered_df.nsmallest(10, 'Rating')
                fig_bottom_10 = plt.figure(figsize=(10, 6))
                sns.barplot(data=bottom_10_movies, x='Rating', y='Title', palette='Reds_d')
                plt.title('Bottom 10 Movies by Rating')
                plt.xlabel('Rating')
                plt.ylabel('Movie Title')
                st.pyplot(fig_bottom_10)

            with st.expander("Release Year Analysis"):
                st.subheader('Movies Distribution by Release Year')
                fig_year_dist = plt.figure(figsize=(13, 6))
                sns.countplot(data=filtered_df, x='Release Year', palette='Set2')
                plt.title('Movies Distribution by Release Year')
                plt.xlabel('Release Year')
                plt.ylabel('Number of Movies')
                plt.xticks(rotation=90)
                st.pyplot(fig_year_dist)

                st.subheader('Year vs Rating')
                fig_year_rating_line = plt.figure(figsize=(12, 6))
                sns.lineplot(data=filtered_df.sort_values('Release Year'), x='Release Year', y='Rating', marker='o', color='orange')
                plt.title('Year vs Rating')
                plt.xlabel('Release Year')
                plt.ylabel('Rating')
                plt.xticks(rotation=45)
                st.pyplot(fig_year_rating_line)

                most_voted_movie = df.loc[df['Votes'].idxmax()]
                least_voted_movie = df.loc[df['Votes'].idxmin()]

                st.subheader('Average Movie Ratings Over Time (Filtered)')
                avg_ratings_per_year_filtered = filtered_df.groupby('Release Year')['Rating'].mean()

                fig_avg_rating_year = plt.figure(figsize=(14, 8))
                sns.lineplot(x=avg_ratings_per_year_filtered.index, y=avg_ratings_per_year_filtered.values, marker='o', color='b')
                plt.title('Average Movie Ratings Over Time (Filtered)')
                plt.xlabel('Release Year')
                plt.ylabel('Average Rating')
                plt.grid(True)
                st.pyplot(fig_avg_rating_year)

            with st.expander("Duration Analysis"):
              

                st.subheader('Duration vs Rating')
                fig_duration_rating_scatter = plt.figure(figsize=(8, 6))
                sns.scatterplot(data=filtered_df, x='Duration', y='Rating', color='purple')
                plt.title('Duration vs Rating')
                plt.xlabel('Duration')
                plt.ylabel('Rating')
                st.pyplot(fig_duration_rating_scatter)

                st.subheader('Votes vs Duration (Filtered)')
                fig_votes_duration_scatter = plt.figure(figsize=(8, 6))
                sns.scatterplot(data=filtered_df, x='Votes', y='Duration', color='brown')
                plt.title('Votes vs Duration (Filtered)', fontsize=15)
                plt.xlabel('Votes', fontsize=12)
                plt.ylabel('Duration (minutes)', fontsize=12)
                st.pyplot(fig_votes_duration_scatter)

                st.subheader('Votes vs Rating with Duration as Bubble Color (Filtered)')
                fig_votes_rating_duration_bubble = plt.figure(figsize=(10, 6))
                sns.scatterplot(x='Votes', y='Rating', hue='Duration', size='Duration', sizes=(10, 200), data=filtered_df, palette='viridis', alpha=0.6)
                plt.title('Votes vs Rating with Duration as Bubble Color (Filtered)')
                plt.xlabel('Votes')
                plt.ylabel('Rating')
                plt.legend(title='Duration', loc='upper right')
                st.pyplot(fig_votes_rating_duration_bubble)

            with st.expander("Votes Analysis"):
               

                st.subheader('Distribution of Movie Votes')
                fig_votes_dist = plt.figure(figsize=(8, 4))
                sns.histplot(filtered_df['Votes'], bins=30, color='red')
                plt.title('Distribution of Movie Votes')
                plt.xlabel('Votes')
                plt.ylabel('Count')
                st.pyplot(fig_votes_dist)

                st.subheader('Votes vs Rating')
                fig_votes_rating_scatter = plt.figure(figsize=(8, 6))
                sns.scatterplot(data=filtered_df, x='Votes', y='Rating', color='green')
                plt.title('Votes vs Rating')
                plt.xlabel('Votes')
                plt.ylabel('Rating')
                st.pyplot(fig_votes_rating_scatter)

                st.subheader('Votes vs Rating with Duration as Bubble Size (Plotly)')
                fig_plotly_votes_rating = px.scatter(
                    filtered_df,
                    x='Votes',
                    y='Rating',
                    size='Duration',
                    color='Genre',
                    hover_name='Title',
                    title='Votes vs Rating with Duration as Bubble Size',
                    size_max=40,
                    height=600
                )
                st.plotly_chart(fig_plotly_votes_rating)

            with st.expander("Genre Analysis"):
                st.subheader('Distribution of Movies by Genre')
                genre_counts = df['Genre'].str.split(',').explode().value_counts()
                fig_genre_dist_bar = plt.figure(figsize=(12, 6))
                sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='Set2')
                plt.title('Distribution of Movies by Genre', fontsize=15)
                plt.xlabel('Genre', fontsize=12)
                plt.ylabel('Number of Movies', fontsize=12)
                plt.xticks(rotation=90)
                st.pyplot(fig_genre_dist_bar)



                st.subheader('Top 10 Movie Genres by Percentage')
                # نستخدم df هنا لحساب نسب الأنواع على البيانات الأصلية
                genre_percentages = df['Genre'].str.split(',').explode().value_counts(normalize=True).head(10)
                fig_genre_percentage_pie = plt.figure(figsize=(8, 8))
                plt.pie(genre_percentages, labels=genre_percentages.index, autopct='%1.1f%%', colors=sns.color_palette('Set3', len(genre_percentages)), startangle=140)
                plt.title('Top 10 Movie Genres by Percentage')
                plt.ylabel('')
                st.pyplot(fig_genre_percentage_pie)

                st.subheader('Average Duration by Genre (Filtered)')
                expanded_filtered_df = filtered_df.copy()
                expanded_filtered_df['Genre'] = expanded_filtered_df['Genre'].str.split(', ')
                expanded_filtered_df = expanded_filtered_df.explode('Genre') # استدعاء explode مرة واحدة فقط
                genre_duration_filtered = expanded_filtered_df.groupby('Genre')['Duration'].mean().sort_values(ascending=False)

                fig_genre_duration_bar = plt.figure(figsize=(12, 6))
                sns.barplot(x=genre_duration_filtered.index, y=genre_duration_filtered.values, palette='magma')
                plt.title('Average Duration by Genre (Filtered)')
                plt.xlabel('Genre')
                plt.ylabel('Average Duration (minutes)')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_genre_duration_bar)



            with st.expander("Correlation Analysis"):
                st.subheader('Correlation Heatmap')
                correlation_matrix = filtered_df[['Release Year', 'Rating', 'Duration', 'Votes']].corr()
                fig_heatmap = plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                plt.title('Correlation Heatmap')
                st.pyplot(fig_heatmap)

        else:
            st.warning("No movies found based on the current filters.")

    # --- Interactive View Tab ---
    with tab3:
        st.title("🎛️ Interactive View")
        st.subheader("Filtered Movie Data")
        st.write("Explore the movie data based on your selections in the sidebar.")
        st.dataframe(filtered_df)

        st.subheader('Compare Rating Distributions by Genre (Filtered)')
        genres_to_compare_interactive = filtered_df['Genre'].str.split(', ').explode().unique()
        if len(genres_to_compare_interactive) >= 2:
            selected_genres_compare_interactive = st.multiselect(
                "Select genres to compare ratings:",
                genres_to_compare_interactive,
                default=genres_to_compare_interactive[:2] if len(genres_to_compare_interactive) >= 2 else genres_to_compare_interactive
            )
            if selected_genres_compare_interactive:
                fig_rating_compare_interactive = plt.figure(figsize=(10, 6))
                for genre in selected_genres_compare_interactive:
                    genre_data = filtered_df[filtered_df['Genre'].str.contains(genre)]['Rating']
                    sns.histplot(genre_data, color=sns.color_palette('viridis')[genres_to_compare_interactive.tolist().index(genre) % len(sns.color_palette('viridis'))], label=genre, kde=True, bins=20)
                plt.legend(title='Genre')
                plt.title('Comparison of Rating Distributions by Selected Genres (Filtered)')
                plt.xlabel('Rating')
                plt.ylabel('Count')
                st.pyplot(fig_rating_compare_interactive)
            else:
                st.info("Select one or more genres to see their rating distributions.")
        else:
            st.info("Not enough genres available based on the current filters to perform a comparison.")


       


if __name__ == "__main__":
    main()