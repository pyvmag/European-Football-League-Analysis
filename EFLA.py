
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import joblib
import os


st.sidebar.header("Dashboard Selector")
dataset_choice = st.sidebar.radio("Choose a Dashboard to Explore", [
                                  "Football Matches", "Player Statistics" ,"Fan Engagement"])
if dataset_choice == "Football Matches":
    # Load the logo using the specified path (replace with correct path)
    logo_path = ""
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=100)

    st.title("Football Match Day Interactive Dashboard")

    # Load the match day data
    @st.cache_data
    def load_match_data():
        data = pd.read_excel("match_data.xlsx", sheet_name="Sheet1")
        data['Date'] = pd.to_datetime(data['Date'])
        return data

    df = load_match_data()

    # Description for Player Statistics
   
    st.subheader("Matchday Statistics Dashboard")
    st.write("""
        Analyze match results with data on home and away teams, goals, fouls, and disciplinary actions (yellow/red cards). 
        Filter matches by league, team, and date range to explore team performance trends and gain insights into specific 
        match outcomes.
    """)

    # Sidebar filters for team data
    st.sidebar.header("Filters")
    with st.sidebar.expander("Select League and Teams", expanded=True):
        selected_league = st.selectbox(
            "🏆 Select League", df['League'].unique(), index=0)
        league_teams = df[df['League'] == selected_league]
        # Define available teams here so it can be used later in the prediction
        available_teams = pd.concat(
            [league_teams['HomeTeam'], league_teams['AwayTeam']]).unique()
        selected_team1 = st.selectbox(
            "⚽ Select Team 1", available_teams, index=0)
        selected_team2 = st.selectbox("⚽ Select Team 2 (optional)", [
                                      "None"] + list(available_teams))
        date_range = st.date_input("📅 Select Date Range", [
                                   df['Date'].min().date(), df['Date'].max().date()])
        date_range = [pd.to_datetime(date) for date in date_range]

    # Filter data based on selections for teams
    filtered_df1 = df[(
        (df['HomeTeam'] == selected_team1) | (df['AwayTeam'] == selected_team1)) &
        (df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])
    ]
    filtered_df2 = None
    if selected_team2 != "None":
        filtered_df2 = df[(
            (df['HomeTeam'] == selected_team2) | (df['AwayTeam'] == selected_team2)) &
            (df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])
        ]

    # Display filtered data
    st.subheader("📊 Filtered Match Data")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Match Data for {selected_team1}**")
        st.write(filtered_df1)
    if filtered_df2 is not None:
        with col2:
            st.write(f"**Match Data for {selected_team2}**")
            st.write(filtered_df2)

    # Function to calculate match insights

    def calculate_insights(filtered_df, team):
        total_goals = filtered_df[filtered_df['HomeTeam'] == team]['HomeGoals'].sum(
        ) + filtered_df[filtered_df['AwayTeam'] == team]['AwayGoals'].sum()
        total_matches = len(filtered_df)
        avg_goals_per_match = total_goals / total_matches if total_matches > 0 else 0
        total_fouls = filtered_df[filtered_df['HomeTeam'] == team]['HomeFouls'].sum(
        ) + filtered_df[filtered_df['AwayTeam'] == team]['AwayFouls'].sum()
        wins = len(filtered_df[(filtered_df['HomeTeam'] == team) & (filtered_df['HomeGoals'] > filtered_df['AwayGoals'])]) + \
            len(filtered_df[(filtered_df['AwayTeam'] == team) & (
                filtered_df['AwayGoals'] > filtered_df['HomeGoals'])])
        losses = len(filtered_df[(filtered_df['HomeTeam'] == team) & (filtered_df['HomeGoals'] < filtered_df['AwayGoals'])]) + len(
            filtered_df[(filtered_df['AwayTeam'] == team) & (filtered_df['AwayGoals'] < filtered_df['HomeGoals'])])
        draws = total_matches - wins - losses
        clean_sheets = len(filtered_df[((filtered_df['HomeTeam'] == team) & (filtered_df['AwayGoals'] == 0)) |
                                       ((filtered_df['AwayTeam'] == team) & (filtered_df['HomeGoals'] == 0))])
        return {
            "Total Matches": total_matches,
            "Wins": wins,
            "Losses": losses,
            "Draws": draws,
            "Clean Sheets": clean_sheets,
            "Total Goals Scored": total_goals,
            "Average Goals per Match": round(avg_goals_per_match, 2),
            "Total Fouls": total_fouls
        }

    # Display match insights side-by-side
    st.subheader("📈 Match Insights Comparison")
    col1, col2 = st.columns(2)
    insights1 = calculate_insights(filtered_df1, selected_team1)
    with col1:
        st.write(f"**Insights for {selected_team1}**")
        for key, value in insights1.items():
            st.metric(label=key, value=value)
    if filtered_df2 is not None:
        insights2 = calculate_insights(filtered_df2, selected_team2)
        with col2:
            st.write(f"**Insights for {selected_team2}**")
            for key, value in insights2.items():
                st.metric(label=key, value=value)

    # Add additional visualizations for top scorers, head-to-head comparison, and goals vs fouls (not included here for brevity)
    def predict_outcome(home_team, away_team):
        # Define the path to the saved model
        model_path = r"match_outcome_predictor.pkl"

    # Check if the model file exists before loading
        if not os.path.exists(model_path):
            print("Model file not found in the expected location.")
            return "Model file missing. Prediction cannot be made."
        else:
            model = joblib.load(model_path)
            print("Model loaded successfully!")

    # Prepare the input data in the same format used during training
        input_data = pd.DataFrame({
            'HomeTeam': [home_team],
            'AwayTeam': [away_team]
        })
        input_data = pd.get_dummies(
            input_data, columns=['HomeTeam', 'AwayTeam'])

    # Add any missing columns to match the training set
        missing_cols = set(model.feature_names_in_) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0

    # Ensure columns are in the same order as the training set
        input_data = input_data[model.feature_names_in_]

    # Make a prediction
        prediction = model.predict(input_data)[0]

    # Calculate win percentages based on historical data
        home_team_data = df[df['HomeTeam'] == home_team]
        away_team_data = df[df['AwayTeam'] == away_team]

        home_team_wins = len(home_team_data[home_team_data['HomeGoals'] > home_team_data['AwayGoals']]) + \
            len(away_team_data[away_team_data['AwayGoals']
                > away_team_data['HomeGoals']])

        away_team_wins = len(home_team_data[home_team_data['AwayGoals'] > home_team_data['HomeGoals']]) + \
            len(away_team_data[away_team_data['HomeGoals']
                > away_team_data['AwayGoals']])

        total_home_matches = len(home_team_data) + len(away_team_data)
        total_away_matches = len(away_team_data) + len(home_team_data)

    # Calculate win percentages
        home_win_percentage = (home_team_wins / total_home_matches) * \
            100 if total_home_matches > 0 else 0
        away_win_percentage = (away_team_wins / total_away_matches) * \
            100 if total_away_matches > 0 else 0

    # Create Horizontal Bar Chart
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh([home_team, away_team], [home_win_percentage,
                away_win_percentage], color=['blue', 'orange'])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Win Percentage (%)")
        ax.set_title(f"Win Percentage Comparison: {home_team} vs {away_team}")

    # Display the Bar Chart in Streamlit
        st.pyplot(fig)

        return prediction, home_win_percentage, away_win_percentage


    # Sidebar for team selection filters
    st.sidebar.header("🔮 Predict Match Outcome")
    home_team = st.sidebar.selectbox(
        "Select Home Team for Prediction", available_teams)
    away_team = st.sidebar.selectbox(
        "Select Away Team for Prediction", available_teams)

    # Prediction button in sidebar
    if st.sidebar.button("Predict Outcome"):
        result, home_win_percentage, away_win_percentage = predict_outcome(
            home_team, away_team)

        # Display the results on the main screen
        st.header("Prediction Result")
        st.write(f"**Predicted Outcome**: {result}")
        st.write(f"**Win Percentage for {home_team}**: {home_win_percentage:.2f}%")
        st.write(f"**Win Percentage for {away_team}**: {away_win_percentage:.2f}%")
    match_data = df

    # Filter data for selected teams
    team1_data = match_data[(match_data['HomeTeam'] == selected_team1) | (
        match_data['AwayTeam'] == selected_team1)]
    team2_data = match_data[(match_data['HomeTeam'] == selected_team2) | (
        match_data['AwayTeam'] == selected_team2)]

    # Head-to-Head Matches
    head_to_head_matches = match_data[
        ((match_data['HomeTeam'] == selected_team1) & (match_data['AwayTeam'] == selected_team2)) |
        ((match_data['HomeTeam'] == selected_team2) & (match_data['AwayTeam'] == selected_team1))
    ]

    # Calculate Wins, Draws, Losses
    team1_wins = len(head_to_head_matches[
        (head_to_head_matches['HomeTeam'] == selected_team1) & (head_to_head_matches['HomeGoals'] > head_to_head_matches['AwayGoals'])
    ]) + len(head_to_head_matches[
        (head_to_head_matches['AwayTeam'] == selected_team1) & (head_to_head_matches['AwayGoals'] > head_to_head_matches['HomeGoals'])
    ])

    team2_wins = len(head_to_head_matches[
        (head_to_head_matches['HomeTeam'] == selected_team2) & (head_to_head_matches['HomeGoals'] > head_to_head_matches['AwayGoals'])
    ]) + len(head_to_head_matches[
        (head_to_head_matches['AwayTeam'] == selected_team2) & (head_to_head_matches['AwayGoals'] > head_to_head_matches['HomeGoals'])
    ])

    draws = len(head_to_head_matches[
        head_to_head_matches['HomeGoals'] == head_to_head_matches['AwayGoals']
    ])

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Wins', 'Draws', 'Losses']
    team1_stats = [team1_wins, draws, len(head_to_head_matches) - team1_wins - draws]
    team2_stats = [team2_wins, draws, len(head_to_head_matches) - team2_wins - draws]

    x = np.arange(len(labels))  # Label locations
    width = 0.35  # Width of the bars

    # Create bars
    rects1 = ax.bar(x - width / 2, team1_stats, width, label=selected_team1, color='blue')
    rects2 = ax.bar(x + width / 2, team2_stats, width, label=selected_team2, color='green')

    # Formatting
    ax.set_ylabel('Count')
    ax.set_title(f'Head-to-Head Comparison: {selected_team1} vs {selected_team2}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    # Display plot
    st.pyplot(fig)


    # Visualizations: Goals and Fouls Comparison for Team 1 and Team 2
    st.subheader("🔍 Goals and Fouls Comparison")

    # Goals Comparison
    col1, col2 = st.columns(2)

    with col1:
        if not filtered_df1.empty:
            fig, ax = plt.subplots()
            home_goals1 = filtered_df1[filtered_df1['HomeTeam']
                                   == selected_team1]['HomeGoals'].sum()
            away_goals1 = filtered_df1[filtered_df1['AwayTeam']
                                   == selected_team1]['AwayGoals'].sum()
            ax.bar(['Home Goals', 'Away Goals'], [home_goals1,
               away_goals1], color=['#1f77b4', '#ff7f0e'])
            ax.set_title(f"{selected_team1} Goals (Bar Chart)")
            ax.set_ylabel("Goals")
            # Dynamic y-axis limit
            ax.set_ylim(0, max(home_goals1, away_goals1) + 1)
            st.pyplot(fig)
        else:
            st.write(f"No data available for {selected_team1}.")

    if filtered_df2 is not None:
        with col2:
            if not filtered_df2.empty:
                fig, ax = plt.subplots()
                home_goals2 = filtered_df2[filtered_df2['HomeTeam']
                                       == selected_team2]['HomeGoals'].sum()
                away_goals2 = filtered_df2[filtered_df2['AwayTeam']
                                       == selected_team2]['AwayGoals'].sum()
                ax.bar(['Home Goals', 'Away Goals'], [home_goals2,
                   away_goals2], color=['#1f77b4', '#ff7f0e'])
                ax.set_title(f"{selected_team2} Goals (Bar Chart)")
                ax.set_ylabel("Goals")
                # Dynamic y-axis limit
                ax.set_ylim(0, max(home_goals2, away_goals2) + 1)
                st.pyplot(fig)
            else:
                st.write(f"No data available for {selected_team2}.")

    # Pie Chart for Fouls Comparison
    st.subheader("⚖️ Fouls Comparison (Pie Chart)")

    col1, col2 = st.columns(2)

    with col1:
        if not filtered_df1.empty:
            fig, ax = plt.subplots()
            home_fouls1 = filtered_df1[filtered_df1['HomeTeam']
                                   == selected_team1]['HomeFouls'].sum()
            away_fouls1 = filtered_df1[filtered_df1['AwayTeam']
                                   == selected_team1]['AwayFouls'].sum()
            ax.pie([home_fouls1, away_fouls1], labels=['Home Fouls', 'Away Fouls'],
               autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
            ax.set_title(f"{selected_team1} Fouls (Pie Chart)")
            st.pyplot(fig)
        else:
            st.write(f"No data available for {selected_team1}.")

    if filtered_df2 is not None:
        with col2:
            if not filtered_df2.empty:
                fig, ax = plt.subplots()
                home_fouls2 = filtered_df2[filtered_df2['HomeTeam']
                                       == selected_team2]['HomeFouls'].sum()
                away_fouls2 = filtered_df2[filtered_df2['AwayTeam']
                                       == selected_team2]['AwayFouls'].sum()
                ax.pie([home_fouls2, away_fouls2], labels=['Home Fouls', 'Away Fouls'],
                   autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
                ax.set_title(f"{selected_team2} Fouls (Pie Chart)")
                st.pyplot(fig)
            else:
                st.write(f"No data available for {selected_team2}.")

# Top Scoring Teams Visualization
    if selected_league:
        st.subheader(
            f"🏆 Top Scoring Teams in the {selected_league} within the selected date range")

    # Filter data based on selected league and date range
        filtered_league_data = df[(df['League'] == selected_league) &
                                  (df['Date'] >= date_range[0]) &
                                  (df['Date'] <= date_range[1])]

    # Calculate total goals for each team in the filtered data
    total_goals = filtered_league_data.groupby(
        'HomeTeam')['HomeGoals'].sum().reset_index()
    away_goals = filtered_league_data.groupby(
        'AwayTeam')['AwayGoals'].sum().reset_index()

    total_goals = total_goals.rename(
        columns={'HomeTeam': 'Team', 'HomeGoals': 'TotalGoals'})
    away_goals = away_goals.rename(
        columns={'AwayTeam': 'Team', 'AwayGoals': 'TotalGoals'})

    # Merge home and away goals
    total_goals = pd.concat([total_goals, away_goals]).groupby(
        'Team', as_index=False).sum()
    top_scorers = total_goals.nlargest(6, 'TotalGoals')

    # Display top scorers
    st.write("### Top 6 Scoring Teams")

    # Bar chart for top scoring teams
    fig, ax = plt.subplots()
    ax.bar(top_scorers['Team'], top_scorers['TotalGoals'], color='green')
    ax.set_xlabel("Teams")
    ax.set_ylabel("Total Goals")
    ax.set_title(f"Top 6 Scoring Teams in the {selected_league}")
    plt.xticks(rotation=45)
    st.pyplot(fig)


elif dataset_choice == "Player Statistics":
    st.title("Player Performance Dashboard")
    import plotly.express as px

    # Load dataset
    df = pd.read_csv('Pset.csv')

    # Remove duplicates
    df = df.drop_duplicates()

    # Ensure 'goals_player' is numeric
    df['goals_player'] = pd.to_numeric(df['goals_player'], errors='coerce')

    # Ensure the 'date_player' column is in datetime format
    df['date_player'] = pd.to_datetime(df['date_player'])

    st.subheader("Player Performance Dashboard")
    st.write("""
        Explore individual player performance with key metrics like goals, assists, key passes, and disciplinary records 
        (yellow/red cards). Compare players, analyze their performance by position, and track their progress over time 
        with visualizations. Each player’s overall rating is calculated based on their contributions to the team.
    """)

    # Sidebar for Player Selection, Compare Mode, and Date Filter
    st.sidebar.header("Player Options")
    compare_mode = st.sidebar.checkbox("Compare Players", help="Select this option to compare multiple players.")
    date_range = st.sidebar.date_input("Select Date Range", [df['date_player'].min().date(), df['date_player'].max().date()])
    date_range = [pd.to_datetime(date) for date in date_range]
    filtered_df = df[(df['date_player'] >= date_range[0]) & (df['date_player'] <= date_range[1])]

    if not compare_mode:
        # Single Player Insights
        selected_player = st.sidebar.selectbox("Select a Player", filtered_df['playerName'].unique())
        player_data = filtered_df[filtered_df['playerName'] == selected_player]

        # Display Single Player Insights
        st.title(f"Insights for {selected_player}")
        st.write("### Basic Player Stats")
        st.write(f"**Goals Scored**: {player_data['goals_player'].sum()}")
        st.write(f"**Own Goals**: {player_data['ownGoals'].sum()}")
        st.write(f"**Expected Goals (xGoals)**: {player_data['xGoals_player'].sum()}")
        st.write(f"**Total Assists**: {player_data['assists'].sum()}")
        st.write(f"**Key Passes**: {player_data['keyPasses'].sum()}")
        st.write(f"**Yellow Cards**: {player_data['yellowCard'].sum()}")
        st.write(f"**Red Cards**: {player_data['redCard'].sum()}")

        # Overall Rating
        st.write("### Overall Rating")
        goals = player_data['goals_player'].sum()
        assists = player_data['assists'].sum()
        key_passes = player_data['keyPasses'].sum()
        x_goals = player_data['xGoals_player'].sum()
        x_assists = player_data['xAssists'].sum()
        yellow_cards = player_data['yellowCard'].sum()
        red_cards = player_data['redCard'].sum()

        positive_score = (goals * 40) + (assists * 25) + (key_passes * 15) + (x_goals * 10) + (x_assists * 10)
        positive_score = min(positive_score / 100, 100)
        penalty = (yellow_cards * 2) + (red_cards * 5)
        overall_rating = max(0, min(positive_score - penalty, 100))

        st.write(f"**Overall Rating**: {round(overall_rating, 2)}/100")

        # Goals and Assists by Position
        st.write("### Performance by Position")
        position_stats = player_data.groupby('position').agg(
            Goals=('goals_player', 'sum'),
            Assists=('assists', 'sum')
        ).reset_index()

        fig = px.bar(
            position_stats,
            x='position',
            y=['Goals', 'Assists'],
            barmode='group',
            color_discrete_map={'Goals': 'green', 'Assists': 'blue'},
            title="Goals and Assists by Position"
        )
        st.plotly_chart(fig)

        # Yellow and Red Cards
        st.write("### Disciplinary Record")
        disciplinary = player_data[['yellowCard', 'redCard']].sum().reset_index()
        disciplinary.columns = ['Card Type', 'Count']

        fig = px.bar(
            disciplinary,
            x='Card Type',
            y='Count',
            color='Card Type',
            color_discrete_map={'yellowCard': 'yellow', 'redCard': 'red'},
            title="Yellow and Red Cards"
        )
        st.plotly_chart(fig)

        # Performance Over Time
        st.write("### Performance Over Time")
        goals_over_time = player_data.groupby('date_player')['goals_player'].sum().reset_index()
        assists_over_time = player_data.groupby('date_player')['assists'].sum().reset_index()

        fig = px.line(
            goals_over_time,
            x='date_player',
            y='goals_player',
            title="Goals Over Time",
            labels={'date_player': 'Date', 'goals_player': 'Goals'},
            line_shape='spline'
        )
        fig.add_scatter(
            x=assists_over_time['date_player'],
            y=assists_over_time['assists'],
            mode='lines+markers',
            name='Assists',
            line=dict(color='blue')
        )
        st.plotly_chart(fig)


    else:
        # Multiple Player Comparison
        st.title("Compare Players")
        selected_players = st.sidebar.multiselect("Select Players to Compare", filtered_df['playerName'].unique())

        if selected_players:
            players_data = filtered_df[filtered_df['playerName'].isin(selected_players)]

            # Compute stats for selected players
            comparison_stats = players_data.groupby('playerName').agg(
                Goals_Scored=('goals_player', 'sum'),
                Assists=('assists', 'sum'),
                Key_Passes=('keyPasses', 'sum'),
                Yellow_Cards=('yellowCard', 'sum'),
                Red_Cards=('redCard', 'sum')
            ).reset_index()

            # Overall Rating
            comparison_stats['Overall_Rating'] = (
                (comparison_stats['Goals_Scored'] * 40) +
                (comparison_stats['Assists'] * 25) +
                (comparison_stats['Key_Passes'] * 15)
            ) / 100 - (comparison_stats['Yellow_Cards'] * 2) - (comparison_stats['Red_Cards'] * 5)
            comparison_stats['Overall_Rating'] = comparison_stats['Overall_Rating'].clip(0, 100)
            comparison_stats['Rank'] = comparison_stats['Overall_Rating'].rank(ascending=False).astype(int)
            comparison_stats = comparison_stats.sort_values('Rank')

            # Display Comparison Table
            st.write("### Player Comparison Stats")
            st.dataframe(comparison_stats)

            # Comparison Graphs
            st.write("### Goals and Assists Comparison")
            fig = px.bar(
                comparison_stats,
                x='playerName',
                y=['Goals_Scored', 'Assists'],
                barmode='group',
                color_discrete_map={'Goals_Scored': 'green', 'Assists': 'blue'},
                title="Goals and Assists Comparison"
            )
            st.plotly_chart(fig)

            st.write("### Disciplinary Record Comparison")
            fig = px.bar(
                comparison_stats,
                x='playerName',
                y=['Yellow_Cards', 'Red_Cards'],
                barmode='group',
                color_discrete_map={'Yellow_Cards': 'yellow', 'Red_Cards': 'red'},
                title="Disciplinary Record Comparison"
            )
            st.plotly_chart(fig)

elif dataset_choice == "Fan Engagement":
    import plotly.express as px
    
# Load dataset
    file_path = "fan_engagement_dataset.csv"  # Ensure this file is in the same directory as the Streamlit app
    df = pd.read_csv(file_path)

    # Define realistic derby and rivalry matches
    derby_rivalry_matches = {
        "Manchester Derby": ["Manchester City", "Manchester United"],
        "El Clasico": ["Real Madrid", "Barcelona"],
        "Milan Derby": ["AC Milan", "Inter Milan"],
        "North London Derby": ["Arsenal", "Tottenham Hotspur"],
        "Merseyside Derby": ["Liverpool", "Everton"],
        "Madrid Derby": ["Real Madrid", "Atletico Madrid"],
        "Rhein Derby": ["Borussia Dortmund", "Schalke 04"]
    }

    # Sidebar Team Selection
    st.sidebar.header("Team Selection")
    selected_teams = st.sidebar.multiselect("Select Teams (Home or Away)", 
                                            options=pd.concat([df["Home Team"], df["Away Team"]]).unique())

    # Submit button to apply filters
    if st.sidebar.button("Submit"):
        if selected_teams:
            # Filter data based on selected teams
            filtered_df = df[(df["Home Team"].isin(selected_teams)) | (df["Away Team"].isin(selected_teams))]

            # Main Dashboard
            st.title("Fan Engagement Dashboard")

            # Overview Metrics
            st.subheader("Overview")
            total_matches = filtered_df.shape[0]
            avg_crowd_strength = int(filtered_df["Crowd Strength"].mean())
            avg_utilization = round(filtered_df["Stadium Capacity Utilization"].mean(), 2)
            st.write(f"Total Matches: {total_matches}")
            st.write(f"Average Crowd Strength: {avg_crowd_strength}")
            st.write(f"Average Stadium Utilization: {avg_utilization}%")

            # Visualizations
            st.subheader("Visualizations")

            # Bar Chart: Crowd Strength by Match Status
            st.write("### Crowd Strength by Match Status")
            status_crowd_chart = px.bar(filtered_df.groupby("Status of Match")["Crowd Strength"].mean().reset_index(),
                                        x="Status of Match", y="Crowd Strength", title="Average Crowd Strength by Match Status")
            st.plotly_chart(status_crowd_chart)

            # Heatmap: Security Personnel vs Crowd Strength
            st.write("### Security Personnel vs Crowd Strength")
            heatmap = px.density_heatmap(filtered_df, x="Crowd Strength", y="Security Personnel Required",
                                        title="Security Personnel vs Crowd Strength", nbinsx=20, nbinsy=20)
            st.plotly_chart(heatmap)

            # Line Chart: Crowd Strength Over Days of the Week
            st.write("### Crowd Strength Over Days of the Week")
            day_crowd_chart = px.line(
                filtered_df.groupby("Day of Week")["Crowd Strength"].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index(),
                x="Day of Week", y="Crowd Strength", title="Average Crowd Strength by Day of the Week")
            st.plotly_chart(day_crowd_chart)

            # Management Summary
            st.subheader("Management Prerequisites Summary")
            for _, match in filtered_df.iterrows():
                st.write(f"**Match:** {match['Home Team']} vs {match['Away Team']}")
                st.write(f"- **Status:** {match['Status of Match']}")
                st.write(f"- **Expected Crowd:** {match['Crowd Strength']} attendees")
                st.write(f"- **Security Needed:** {match['Security Personnel Required']} personnel")
                st.write(f"- **Weather:** {match['Weather']}")
                st.write(f"- **Average Ticket Price:** ${match['Ticket Price']}")
                st.write("---")

        else:
            st.warning("Please select at least one team to view the data and visualizations.")

    else:
        st.title("Fan Engagement Dashboard")
        st.write("Use the sidebar to select teams and hit Submit to view visualizations, statistics, and management insights.")
