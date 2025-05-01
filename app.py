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
    return joblib.load("imdb_data.joblib")

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
    filtered_df = filtered_df[(filtered_df['Rating'] >= min_r) & (filtered_df['Rating'] <= max_r)]

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

        col_tables1, col_tables2 = st.columns(2)

        with col_tables1:
            st.subheader("Top 5 Highest Rated Movies (Filtered)")
            top_5_rated = filtered_df.nlargest(5, 'Rating')[['Title', 'Rating', 'Release Year']]
            st.dataframe(top_5_rated)

            st.subheader("5 Most Recent Movies (Filtered)")
            most_recent_5 = filtered_df.sort_values('Release Year', ascending=False).head(5)[['Title', 'Release Year', 'Rating']]
            st.dataframe(most_recent_5)

        with col_tables2:
            st.subheader("Summary Statistics (Filtered)")
            summary_stats = filtered_df[['Rating', 'Duration', 'Votes']].describe().T
            st.dataframe(summary_stats)

            st.subheader("Top 5 Most Frequent Genres (Filtered)")
            genres_exploded_filtered = filtered_df['Genre'].str.split(', ').explode()
            top_5_genres = genres_exploded_filtered.value_counts().nlargest(5).reset_index()
            top_5_genres.columns = ['Genre', 'Count']
            st.dataframe(top_5_genres)

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

                st.subheader('Rating Distribution (KDE)')
                fig_kde_rating = plt.figure(figsize=(10, 5))
                sns.kdeplot(filtered_df['Rating'], fill=True, color="skyblue")
                plt.title('Distribution of Ratings')
                plt.xlabel('Rating')
                plt.ylabel('Density')
                st.pyplot(fig_kde_rating)

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
                sns.countplot(data=filtered_df, x='Release Year', palette='viridis')
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

            with st.expander("Duration Analysis"):
                st.subheader('Distribution of Movie Duration')
                fig_duration_dist = plt.figure(figsize=(8, 6))
                sns.histplot(filtered_df['Duration'], bins=20, kde=True, color='teal', edgecolor='black')
                plt.title('Distribution of Movie Duration')
                plt.xlabel('Duration (minutes)')
                plt.ylabel('Frequency')
                st.pyplot(fig_duration_dist)

                st.subheader('Duration vs Rating')
                fig_duration_rating_scatter = plt.figure(figsize=(8, 6))
                sns.scatterplot(data=filtered_df, x='Duration', y='Rating', color='purple')
                plt.title('Duration vs Rating')
                plt.xlabel('Duration')
                plt.ylabel('Rating')
                st.pyplot(fig_duration_rating_scatter)

            with st.expander("Votes Analysis"):
                st.subheader('Distribution of Movie Votes (Log Scale)')
                fig_votes_dist_log = plt.figure(figsize=(8, 4))
                sns.histplot(filtered_df['Votes'], bins=30, log_scale=True, color='red')
                plt.title('Distribution of Movie Votes (Log Scale)')
                plt.xlabel('Votes')
                plt.ylabel('Count')
                st.pyplot(fig_votes_dist_log)

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
                genres_exploded = filtered_df['Genre'].str.split(', ').explode().reset_index(drop=True)
                ratings_reset = filtered_df['Rating'].reset_index(drop=True)

                avg_rating_by_genre = pd.DataFrame({
                    'Genre': genres_exploded,
                    'Rating': ratings_reset
                }).groupby('Genre')['Rating'].mean().sort_values(ascending=False)

                fig_avg_rating_genre_bar = plt.figure(figsize=(30, 6))
                sns.barplot(x=avg_rating_by_genre.index, y=avg_rating_by_genre.values, palette='coolwarm')
                plt.title('Average Rating by Genre')
                plt.xlabel('Genre')
                plt.ylabel('Average Rating')
                plt.xticks(rotation=90)
                st.pyplot(fig_avg_rating_genre_bar)

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

        st.subheader("Rating Distribution (Filtered)")
        fig_rating_dist_filtered = plt.figure(figsize=(8, 6))
        sns.histplot(filtered_df['Rating'], bins=20, kde=True, color='orange', edgecolor='black')
        plt.title('Distribution of Movie Ratings (Filtered)')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        st.pyplot(fig_rating_dist_filtered)

        # You can add more interactive plots or tables here that directly reflect the sidebar filters
        # For example, a scatter plot of 'Votes' vs 'Rating' for the filtered data:
        st.subheader("Votes vs Rating (Filtered)")
        fig_votes_rating_scatter_filtered = plt.figure(figsize=(8, 6))
        sns.scatterplot(data=filtered_df, x='Votes', y='Rating', color='purple')
        plt.title('Votes vs Rating (Filtered)')
        plt.xlabel('Votes')
        plt.ylabel('Rating')
        st.pyplot(fig_votes_rating_scatter_filtered)

if __name__ == "__main__":
    main()